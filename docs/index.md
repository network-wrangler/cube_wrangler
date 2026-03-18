# Cube Wrangler

Cube Wrangler is a Python package that provides utilities for bridging [Network Wrangler](https://network-wrangler.github.io/network_wrangler/) with [Bentley Cube](https://www.bentley.com/software/cube/), a commercial travel modeling software.

## Capabilities

- **Roadway export**: Creates files that can be read by a Cube script to build a Cube roadway network from a Network Wrangler `RoadwayNetwork`.
- **Transit export**: Writes a Network Wrangler `TransitNetwork` in Cube `.lin` format.
- **Log → Project Card**: Converts a Cube Log file (a record of roadway edits made in Cube) into a [Project Card](https://github.com/network-wrangler/projectcard).
- **LIN diff → Project Card**: Compares two Cube `.lin` files and creates a Project Card representing the transit edits (Cube does not log transit edits; it rewrites the full `.lin` file).

## Installation

Cube Wrangler requires Python 3.10 or later.

### Using uv (recommended)

```bash
uv add cube-wrangler
```

### Using pip

```bash
pip install cube-wrangler
```

### Development install

```bash
git clone https://github.com/network-wrangler/cube_wrangler
cd cube_wrangler
uv sync
```

## Quick Start

### Convert a Cube Log file to Project Cards

```python
from cube_wrangler.project import Project
import network_wrangler as nw

# Load your base network
net = nw.load_roadway_from_dir("my_network/")

# Convert Cube log to a project card
project = Project.create_project(
    base_roadway_network=net,
    roadway_log_file="changes.log",
)
project.write_project_card("output/")
```

### Export Network Wrangler network to Cube

```python
from cube_wrangler.roadway import StandardRoadway

std = StandardRoadway(net, parameters=params)
std.write_cube_net("output/")
```

## Ecosystem

Cube Wrangler is part of the Network Wrangler ecosystem:

```
projectcard  →  network_wrangler  →  cube_wrangler
(schema)        (applies cards)      (diffs Cube files, emits cards)
```

| Package | Role |
|---|---|
| [projectcard](https://github.com/network-wrangler/projectcard) | Project Card schema and validation |
| [network_wrangler](https://github.com/network-wrangler/network_wrangler) | Core network manipulation |
| [cube_wrangler](https://github.com/network-wrangler/cube_wrangler) | Cube ↔ Network Wrangler bridge |
