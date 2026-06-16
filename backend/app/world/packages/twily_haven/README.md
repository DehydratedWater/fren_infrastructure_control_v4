# Twily Haven

The authored world package for **Twily** — her private inner life as a
scholar-tinkerer in the cozy, faintly-enchanted canal town of **Mooring Wells**.
The engine loads and merges the YAML files in this directory into one
`WorldPackage` (see `backend/app/world/models.py`) and validates it.

## Files

| File             | What it holds                                                        |
|------------------|----------------------------------------------------------------------|
| `package.yaml`   | Top-level world: id/name/version/description, `setting`, `protagonist` (Twily), `visitor` (Vis), `factions`, `scenario`. |
| `locations.yaml` | `locations` (home rooms + town places) and the `connections` graph.  |
| `npcs.yaml`      | `npcs` — the town's cast.                                            |
| `lorebook.yaml`  | `lorebook` — keyword-triggered world flavour.                        |

Each top-level key maps to a field on `WorldPackage`. Keep them split as above;
the loader concatenates the lists.

## Editing rules (schema is strict)

- **Match field names exactly** and add no extra fields — most models use
  `extra="forbid"`, so a typo'd or unknown key fails validation.
- **Referential integrity** — every id referenced must resolve:
  - `scenario.starting_location_id` → a real `locations[].id`
  - every `connections[].from_id` / `to_id` → real location ids
  - every `locations[].default_npcs[]` → real `npcs[].id`
  - every `npcs[].home_location_id` → real location id
  - `locations[].parent_id` → a real location id (or `null`)
- **One connected graph** — Twily must be able to walk from any room to any town
  place. Add new places by giving them a `connections` edge to an existing node
  (usually `canal_street` or `towpath`).
- **The study must keep an activity with `tag: computer`** — the engine routes
  Twily's web-research through it. Other special tags the engine keys on:
  `rest` (sleep) and `cook`.
- **NPC presence**: set an NPC's `home_location_id` *and* add their id to that
  location's `default_npcs` so they show up as present.

## Activity tags in use

`computer`, `tinker`, `read`, `journal`, `rest`, `cook`, `brew_tea`,
`tend_plants`, `think`, `refresh`, `depart`, `walk`, `socialize`,
`consult_index`, `collaborate`, `browse`, `shop`. Reuse these where sensible so
behaviour stays consistent.

## Extending the world

- **New room**: add to `locations` with `parent_id: twily_home`, connect it to
  `twily_hall` (or another room) in `connections`.
- **New town place**: add with `parent_id: null`, connect to `canal_street` or
  `towpath`, give it 2–3 activities and (optionally) a resident NPC.
- **New NPC**: add to `npcs.yaml`, set a real `home_location_id`, and list them
  in that location's `default_npcs`.
- **New lore**: add a `lorebook` entry with distinctive `keywords` and a
  `priority` (higher wins when context budget is tight).

Tone to preserve: cozy, literate, a little whimsical, emotionally real. Magic is
mundane infrastructure, never spectacle. Small stakes — a recipe, a friendship,
a stubborn circuit. Snark over a warm heart.
