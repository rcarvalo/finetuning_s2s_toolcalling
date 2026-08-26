"""Bridge to Inspect AI: our runs become Inspect logs, its viewer plays our audio.

Why adopt Inspect rather than write a viewer: its log viewer already renders an
``<audio controls>`` player for any message carrying ``ContentAudio`` — verified
in its bundled front-end, where the content renderer maps ``audio`` to a real
audio element. One condition: the source must be a base64 data URI, since
``isRenderableAudioSource`` checks the MIME type and a bare path degrades to a
plain reference. Hence :mod:`lfm2_audio.inspect_bridge.audio`.

The package is named ``inspect_bridge`` and not ``inspect``: the latter would
shadow the standard library module for every import in the process.

``inspect_ai`` is an optional dependency (extra ``inspect``): nothing here is
imported unless someone exports a log.
"""
