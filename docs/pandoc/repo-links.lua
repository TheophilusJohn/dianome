-- pandoc Lua filter: relative repo links in docs/writeup.md (`../docs/phase-4-notes.md#cost-harness`) become GitHub
-- blob URLs at main, so the PDF's citation links resolve. External links are left alone.
local BLOB = "https://github.com/TheophilusJohn/dianome/blob/main/"
function Link(el)
  local t = el.target
  if t:match("^%a+://") or t:match("^#") then return el end
  t = t:gsub("^%.%./", "")
  el.target = BLOB .. t
  return el
end
