// The export's own check that the main video is in the finished file (python/picture_check.py): on
// QA (2026-11-05) an export came out with the speaker black from one part to the end, sound fine,
// FFmpeg reporting success. A black stretch in the video itself must not count as missing.
// Run: npx tsx --test src/jobs/picture_check.test.ts (needs FFmpeg and Python)
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

const PY = fileURLToPath(new URL('../python/', import.meta.url))
const python = (script: string) =>
  JSON.parse(execFileSync(process.platform === 'win32' ? 'python' : 'python3', ['-c', script], { encoding: 'utf8' }))

test('a part whose video is missing is found; one black in the video itself is not', () => {
  const dir = mkdtempSync(join(tmpdir(), 'pcheck-'))
  try {
    const ff = (...args: string[]) => execFileSync('ffmpeg', ['-v', 'error', '-y', ...args])
    const src = join(dir, 'src.mp4'), good = join(dir, 'good.mp4'), broken = join(dir, 'broken.mp4')
    // The video: a picture for 3 s, then black for 3 s
    ff('-f', 'lavfi', '-i', 'testsrc2=s=216x384:r=30:d=3', '-f', 'lavfi', '-i', 'color=c=black:s=216x384:r=30:d=3',
      '-filter_complex', '[0:v][1:v]concat=n=2:v=1[v]', '-map', '[v]', '-pix_fmt', 'yuv420p', src)
    // A right export (the same), and one that lost the video (black throughout)
    ff('-i', src, '-c', 'copy', good)
    ff('-f', 'lavfi', '-i', 'color=c=black:s=216x384:r=30:d=6', '-pix_fmt', 'yuv420p', broken)
    const out = python(`
import json, sys
sys.path.insert(0, ${JSON.stringify(PY)})
from picture_check import main_picture_probes, missing_main_picture
box = lambda s: [{"slot_index": 0, "source_offset_ms": s}]
segs = [{"start_ms": 0, "end_ms": 3000, "layout": "vertical", "crop_boxes": box(0)},
        {"start_ms": 3000, "end_ms": 6000, "layout": "vertical", "crop_boxes": box(3000)}]
probes = main_picture_probes(segs, 216, 384, [], lambda s: True)
miss = lambda f: [p["t"] for p in missing_main_picture(f, probes, 216, 384, ${JSON.stringify(src)}, 0.0)]
# A frame: a photo in slot 0, the main video in slot 1 (Pic + Vid): only slot 1 is looked at
frame = {"start_ms": 0, "end_ms": 4000, "layout": "frame_screenshot", "crop_boxes": [],
         "frame": {"main_slots": [1], "items": [{"id": "p", "kind": "photo", "lane": 0, "start_ms": 0, "end_ms": 4000, "image_path": "x.jpg"}]}}
fp = main_picture_probes([frame], 1080, 1920, [], lambda s: True)
# Hidden stretches (black on purpose) aren't looked at
hidden = main_picture_probes(segs, 216, 384, [(0, 6000)], lambda s: True)
print(json.dumps({"looks": [p["t"] for p in probes], "good": miss(${JSON.stringify(good)}), "broken": miss(${JSON.stringify(broken)}),
                  "frame_slots": [p.get("slot") for p in fp], "hidden": len(hidden)}))
`)
    assert.deepEqual(out.looks, [1.5, 4.5, 5])
    assert.deepEqual(out.good, [])
    // (4.5 s and 5 s are black in the video itself: not missing)
    assert.deepEqual(out.broken, [1.5])
    // (its middle, and near its end: it is the clip's last part)
    assert.deepEqual(out.frame_slots, [1, 1])
    assert.equal(out.hidden, 0)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
