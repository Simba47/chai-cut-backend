import {
  S3Client, GetObjectCommand, PutObjectCommand,
  CreateMultipartUploadCommand, UploadPartCommand, CompleteMultipartUploadCommand, AbortMultipartUploadCommand,
} from '@aws-sdk/client-s3'
import { getSignedUrl } from '@aws-sdk/s3-request-presigner'
import { createReadStream, createWriteStream } from 'node:fs'
import { open, stat } from 'node:fs/promises'
import { pipeline } from 'node:stream/promises'
import { Readable } from 'node:stream'

const endpoint = process.env.R2_ENDPOINT
const accessKeyId = process.env.R2_ACCESS_KEY_ID
const secretAccessKey = process.env.R2_SECRET_ACCESS_KEY

if (!endpoint || !accessKeyId || !secretAccessKey) {
  throw new Error('R2_ENDPOINT, R2_ACCESS_KEY_ID, and R2_SECRET_ACCESS_KEY must be set')
}

export const r2 = new S3Client({
  region: 'auto',
  endpoint,
  credentials: { accessKeyId, secretAccessKey },
  requestChecksumCalculation: 'WHEN_REQUIRED',
  responseChecksumValidation: 'WHEN_REQUIRED',
})

export const R2_BUCKET = process.env.R2_BUCKET ?? 'chai-cut-videos'

// For the parts of an upload in parts: each part is sent as a stream so the bytes can be counted as they
// go, and a stream can't be sent twice — so this client doesn't try again itself; r2UploadFileInParts does
const r2Parts = new S3Client({
  region: 'auto',
  endpoint,
  credentials: { accessKeyId, secretAccessKey },
  requestChecksumCalculation: 'WHEN_REQUIRED',
  responseChecksumValidation: 'WHEN_REQUIRED',
  maxAttempts: 1,
})

// Videos can be several GB: move them between R2 and disk as streams, never as one in-memory
// Buffer (that ran the worker out of memory and took every job running with it down too)

/** Stream an R2 object straight to a local file */
export async function r2DownloadToFile(key: string, filePath: string, signal?: AbortSignal): Promise<void> {
  const res = await r2.send(new GetObjectCommand({ Bucket: R2_BUCKET, Key: key }), { abortSignal: signal })
  await pipeline(Readable.from(res.Body as AsyncIterable<Uint8Array>), createWriteStream(filePath), { signal })
}

/**
 * A temporary link to an R2 object (for `seconds`). FFmpeg reads a video through it with range
 * requests, fetching only the part it seeks to — not the whole file.
 */
export async function r2SignedUrl(key: string, seconds = 2 * 60 * 60): Promise<string> {
  return getSignedUrl(r2, new GetObjectCommand({ Bucket: R2_BUCKET, Key: key }), { expiresIn: seconds })
}

/** Stream a local file up to R2 */
export async function r2UploadFile(key: string, filePath: string, contentType: string): Promise<void> {
  // A local export (src/scripts/local-export.ts) keeps the file on the PC instead of uploading it
  if (process.env.LOCAL_EXPORT_DIR) {
    const { copyFile, mkdir } = await import('node:fs/promises')
    const { join } = await import('node:path')
    await mkdir(process.env.LOCAL_EXPORT_DIR, { recursive: true })
    const dest = join(process.env.LOCAL_EXPORT_DIR, key.replace(/[\\/]+/g, '_'))
    await copyFile(filePath, dest)
    console.log(`[local export] saved ${dest} (not uploaded)`)
    return
  }
  await r2.send(new PutObjectCommand({ Bucket: R2_BUCKET, Key: key, Body: createReadStream(filePath), ContentType: contentType }))
}

/** Parts a big file goes up in (R2: every part but the last the same size, at least 5 MiB, at most 10,000 parts) */
export const UPLOAD_PART_BYTES = 8 * 1024 * 1024
const UPLOAD_PARTS_AT_ONCE = 4

/**
 * Upload a local file to R2 in parts, saying how much is up after each part (`onProgress`) — what a link
 * import's dashboard bar shows while the video is saved (user, 2026-11-12: one single upload left the bar
 * still for the whole save). Four parts go up at once, read from disk one part each (never the whole
 * file in memory); a part that fails is tried again (up to 3 times), and an upload that fails or is stopped
 * is cancelled on R2 so no half-made file is left. A file of one part goes up as before.
 */
export async function r2UploadFileInParts(
  key: string, filePath: string, contentType: string,
  opts: { onProgress?: (done: number, total: number) => void; signal?: AbortSignal } = {},
): Promise<void> {
  const { onProgress, signal } = opts
  const { size } = await stat(filePath)
  if (process.env.LOCAL_EXPORT_DIR || size <= UPLOAD_PART_BYTES) {
    await r2UploadFile(key, filePath, contentType)
    onProgress?.(size, size)
    return
  }
  const MiB = 1024 * 1024
  const partBytes = Math.max(UPLOAD_PART_BYTES, Math.ceil(size / 9000 / MiB) * MiB)
  const count = Math.ceil(size / partBytes)

  const { UploadId } = await r2.send(
    new CreateMultipartUploadCommand({ Bucket: R2_BUCKET, Key: key, ContentType: contentType }), { abortSignal: signal })
  if (!UploadId) throw new Error('R2 did not start the upload of ' + key)

  const file = await open(filePath, 'r')
  const etags: string[] = new Array(count)
  let next = 0
  let done = 0
  let failed = false
  const lane = async () => {
    while (!failed && next < count) {
      signal?.throwIfAborted()
      const i = next++
      const start = i * partBytes
      const length = Math.min(partBytes, size - start)
      const body = Buffer.allocUnsafe(length)
      for (let got = 0; got < length;) {
        const { bytesRead } = await file.read(body, got, length - got, start + got)
        if (!bytesRead) throw new Error(`${filePath} ended before ${size} bytes`)
        got += bytesRead
      }
      etags[i] = await sendPart(i + 1, body)
    }
  }
  // One part, its bytes counted as they leave (the bar moves while the parts go up, not only when one
  // ends: four ending together moved it in big jumps); tried up to 3 times, its count taken back on a failure
  const sendPart = async (partNumber: number, body: Buffer): Promise<string> => {
    for (let attempt = 1; ; attempt++) {
      let sent = 0
      const stream = Readable.from((function* () {
        for (let at = 0; at < body.length; at += 256 * 1024) {
          const piece = body.subarray(at, at + 256 * 1024)
          sent += piece.length
          done += piece.length
          onProgress?.(Math.min(done, size), size)
          yield piece
        }
      })())
      try {
        const res = await r2Parts.send(new UploadPartCommand({
          Bucket: R2_BUCKET, Key: key, UploadId, PartNumber: partNumber, Body: stream, ContentLength: body.length,
        }), { abortSignal: signal })
        if (!res.ETag) throw new Error(`R2 gave no ETag for part ${partNumber} of ${key}`)
        return res.ETag
      } catch (e) {
        done -= sent
        if (signal?.aborted || failed || attempt >= 3) throw e
        await new Promise(r => setTimeout(r, 500 * attempt))
      }
    }
  }
  try {
    // every lane is waited for (none left writing after a failure), then the first failure is reported
    const lanes = await Promise.allSettled(Array.from({ length: Math.min(UPLOAD_PARTS_AT_ONCE, count) }, () =>
      lane().catch(e => { failed = true; throw e })))
    const failure = lanes.find(l => l.status === 'rejected')
    if (failure) throw (failure as PromiseRejectedResult).reason
    await r2.send(new CompleteMultipartUploadCommand({
      Bucket: R2_BUCKET, Key: key, UploadId,
      MultipartUpload: { Parts: etags.map((ETag, i) => ({ ETag, PartNumber: i + 1 })) },
    }), { abortSignal: signal })
  } catch (e) {
    await r2.send(new AbortMultipartUploadCommand({ Bucket: R2_BUCKET, Key: key, UploadId })).catch(() => {})
    throw e
  } finally {
    await file.close()
  }
}
