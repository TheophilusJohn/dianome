# Spike 00 — storage partitioning

Harness for `docs/spikes/00-storage-partitioning.md`. Vanilla HTML/JS, no
build step. Three static sites:

| Directory | Cloudflare Pages project | Role |
| --- | --- | --- |
| `cdn/` | `dianome-cdn` | owns the cache: `frame.html`, `harness.js`, `probe.js`, `chunks/test.bin` |
| `site-a/` | `dianome-spike-a` | embedding site A |
| `site-b/` | `dianome-spike-b` | embedding site B |

The CDN origin (`https://dianome-cdn.pages.dev`) is a constant at the top of
`cdn/harness.js` and in the `<script src>` of `site-a/index.html` and
`site-b/index.html`. Change it in those three places if the project name is
taken.

## 1. Generate the payload and record its hash

```sh
cd spikes/00-storage
head -c 20971520 /dev/urandom > cdn/chunks/test.bin && sha256sum cdn/chunks/test.bin
```

20 MB of random bytes: Cloudflare Pages rejects files over 25 MiB, and random
data cannot be compressed or deduplicated. `test.bin` is git-ignored; commit
the SHA-256 into the header line of `docs/spikes/00-storage-partitioning.md`.
(On macOS without coreutils use `shasum -a 256`.)

## 2. Deploy the three Pages projects

```sh
cd spikes/00-storage
npx wrangler pages deploy cdn    --project-name dianome-cdn
npx wrangler pages deploy site-a --project-name dianome-spike-a
npx wrangler pages deploy site-b --project-name dianome-spike-b
```

Pages project names are global. If `dianome-cdn` is taken, pick another name
and change the CDN constant in `cdn/harness.js`, `site-a/index.html` and
`site-b/index.html` before deploying. `cdn/_headers` sets the caching and CORS
rules for `/chunks/*`, `/harness.js` and `/probe.js`; nothing sets
`X-Frame-Options`.

## 3. Day-one checks before any measurement

1. `pages.dev` must be on the Public Suffix List, otherwise the three origins
   are one site and every result is meaningless:

   ```sh
   curl -s https://publicsuffix.org/list/public_suffix_list.dat | grep -n 'pages.dev'
   ```

   Record Y/N in the results header.
2. On site A in Chrome: DevTools → Application → Storage. Confirm the frame's
   storage (the `dianome-cdn.pages.dev` entry) shows a partition key. Record
   Y/N in the results header.
3. In the frame's console (select the `frame.html` context in DevTools):
   `typeof document.requestStorageAccess`, and, from a click inside the frame,
   `document.requestStorageAccess({all: true})`. Record whether it returns a
   handle with `.caches`, throws, or prompts, plus the Chrome version.

## 4. Run order per browser

1. Visit `https://dianome-cdn.pages.dev` top-level and click
   **Enable cross-site model cache**.
2. Open site A (`https://dianome-spike-a.pages.dev`), run **T1** in the frame.
3. Open site B (`https://dianome-spike-b.pages.dev`), run **T1**, **T2**,
   **T3**, **T4**, **Probe**, **T5** (T5 last — it fills storage).
4. Close and reopen the browser, rerun **T2** / **T3** on B.
5. Click **Copy results as Markdown** and paste into
   `docs/spikes/00-storage-partitioning.md`. Note by hand whether a prompt was
   shown and how many gestures were needed; the harness cannot detect prompts.
6. Chrome only: repeat steps 1–5 a second time in a profile with
   **Block third-party cookies** enabled (an Incognito window is enough).
   Record it as a separate row, **Chrome (3PC blocked)**, in both results
   tables.

## Local check (no deploy)

Static-serve `cdn/` on one port and `site-a/` on another, with the CDN
constant temporarily pointed at the local `cdn/` server (e.g.
`http://localhost:8000`) in `cdn/harness.js` and `site-a/index.html`. The
`cdn/` server must send `Access-Control-Allow-Origin: *` on `harness.js` and
`probe.js`, because `python3 -m http.server` does not: `npx wrangler pages dev
cdn` honours `_headers`, or wrap `http.server` with a handler that adds the
header. Locally the two ports are the same site, so nothing is partitioned;
this only checks that the pages load and T1 logs a row. Revert the constant
before deploying.
