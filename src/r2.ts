import { S3Client, GetObjectCommand, PutObjectCommand } from '@aws-sdk/client-s3'
import { createReadStream, createWriteStream } from 'node:fs'
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

// Videos can be several GB: move them between R2 and disk as streams, never as one in-memory
// Buffer (that ran the worker out of memory and took every job running with it down too)

/** Stream an R2 object straight to a local file */
export async function r2DownloadToFile(key: string, filePath: string, signal?: AbortSignal): Promise<void> {
  const res = await r2.send(new GetObjectCommand({ Bucket: R2_BUCKET, Key: key }), { abortSignal: signal })
  await pipeline(Readable.from(res.Body as AsyncIterable<Uint8Array>), createWriteStream(filePath), { signal })
}

/** Stream a local file up to R2 */
export async function r2UploadFile(key: string, filePath: string, contentType: string): Promise<void> {
  await r2.send(new PutObjectCommand({ Bucket: R2_BUCKET, Key: key, Body: createReadStream(filePath), ContentType: contentType }))
}
