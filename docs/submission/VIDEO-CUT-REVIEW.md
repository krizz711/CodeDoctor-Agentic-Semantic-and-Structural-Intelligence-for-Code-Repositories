# Local demo cut review — 2026-10-01

This is an internal review note for a local preview. The original recording and
the published YouTube video have not been changed. It is not approval to amend
the submitted demo.

The original template's checklist (slide 11) limits the demo to five minutes.
The original recording is **346.133 seconds**; the local preview is **293.410
seconds (4:53.410)** including its audio stream. The original 2880 × 1800
resolution, 60 fps and stereo audio are retained. There is no playback speed
change, replacement narration, generated footage, added overlay or alteration of the
displayed runtime measurements.

## Original-timeline cuts

These intervals are removed from the original recording, not from an already
trimmed timeline. Timestamps are provisional ASR-derived edit boundaries and
need human listening before publication.

| Removed interval | Duration | Content and reason |
| --- | ---: | --- |
| 00:15.320–00:42.720 | 27.400 s | Introductory benchmark-versus-repository discussion before the architecture explanation. Team identification is retained; the later explanation of APPS-specific execution and its limits is retained. |
| 04:24.760–04:29.960 | 5.200 s | Navigation apology while switching views through a black screen. No query submission, response or computation timer is removed. |
| 05:26.000–05:46.133 | 20.133 s | Closing remarks and extra README scrolling after the repository reference. |

Kept intervals: **00:00.000–00:15.320**, **00:42.720–04:24.760** and
**04:29.960–05:26.000**. All recorded live query/response demonstrations and
the discussion of possible model exposure to APPS are retained. The clip still
shows the recorded historical implementation; trimming cannot update its code
or evidence.

## Files and checks

- Original: `/Users/manishharigovind/Movies/submisssion_vedio.mov`.
- Local preview: `build/local-correction/video/VITV_Doctors_Demo_cut_preview.mp4`.
- Exact cut intervals, hashes and sampled audio checks: `build/local-correction/video/cut-review.json`.
- Probe: `build/local-correction/video/preview-probe.json`.
- Full decode check: `build/local-correction/video/decode-check.log` (exit 0, no reported errors).
- Source/preview cut-boundary contact sheets: `build/local-correction/video/frames/`.
- Provisional transcripts and timestamped token records: `build/local-correction/video/transcript*.json`.
- Render and transcription logs, including the initial sandbox-network failure,
  remain under `build/local-correction/video/`.

The preview has H.264 video and AAC stereo audio; their durations differ by
**10 ms**. Five retained-audio spot checks match the source with correlations
above **0.998** and at most **10 ms** offset. Original and preview boundary
frames were inspected. Those checks confirm encoding and sampled timeline
correspondence; they do not certify every spoken word or complete audiovisual
quality. The preview is about **445 MiB** and is intentionally outside Git.

Whisper tiny.en at revision `87c7102498dcde7456f24cfd30239ca606ed9063`
produced provisional ASR locally. Audio was not sent to a transcription service.
The transcript contains recognition errors and approximate offsets. Still-frame
previews were inspected in Codex.

## Human review before any change to the published demo

Listen to the entire preview and especially the joins at **00:15.320** and
**03:57.360**, plus its final sentence. Confirm that words are not clipped,
the meaning is preserved, audio stays synchronized and displayed results remain
readable. Confirm that the current YouTube timeline matches this local original
before using the coordinates in YouTube Studio. A separately authorized Studio
trim can retain the existing URL; uploading this MP4 as a replacement would not.

The recording repeatedly shows “Searchable — still encoding” and a dense pass
that did not run. It demonstrates lexical/structural lookup, an execution
attempt, historical screening evidence and version comparison. It must not be
described as proof of live 1.7B dense CPU latency or an edit-to-rebuild timing.
Those claims need their separate measured fixtures. No unseen-accuracy result
is established by this recording.

Samsung's permission and the user's explicit authorization are required before
changing any published submission material. This local preview does not make
the existing 5:46 published video comply with the duration limit.
