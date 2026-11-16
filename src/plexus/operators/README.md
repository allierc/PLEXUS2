# Operators

Name, kind, number of `model:` / `implementation:` variants, aliases, title; generated from the registry.

### adhesion_ops

| name | kind | models / impls | aliases | title |
|---|---|---|---|---|
| `contact_finder` | rewire | 0 / 3 |  | detect cell contacts |
| `bond_binding` | lateral | 0 / 3 |  | bind and unbind adhesion |
| `bond_springs` | exchange | 0 / 2 |  | pull adhered cells together |
| `cell_pull` | lateral | 0 / 1 |  | pull cells with a probe |
| `state_dump` | lateral | 0 / 1 |  | save the current state for the live view |
| `lumen_pressure` | exchange | 1 / 4 |  | push each cell's inner face out to its share of the lumen volume |

### bm_shell_ops

| name | kind | models / impls | aliases | title |
|---|---|---|---|---|
| `bm_shell` | lateral | 0 / 1 |  | a stretchable, bendable basement membrane |
| `bm_cell_contact` | exchange | 0 / 2 |  | a basement membrane and the cells inside it |
| `bm_cell_apply` | lateral | 0 / 1 |  | the cells' load on the basement membrane |

### cell_ops

| name | kind | models / impls | aliases | title |
|---|---|---|---|---|
| `aggregate_centroid` | aggregate | 0 / 1 |  | track cell centres |
| `seed_state_from_file` | seed | 0 / 1 |  | load the initial state from a file |
| `seed_state` | seed | 0 / 1 | `seed_state_random` | set the initial state by a rule |
| `laplacian_diffusion` | lateral | 0 / 1 |  | diffuse a value between neighbours |
| `protrusion` | lateral | 1 / 1 |  | push the front and pull the rear |
| `phase_clock` | lateral | 0 / 1 |  | advance each cell's phase clock |
| `polar_active_stress` | lateral | 1 / 1 |  | extend and retract along the polarity |
| `deform_control` | lateral | 0 / 4 |  | change rest shape at a set rate |

### chain_ops

| name | kind | models / impls | aliases | title |
|---|---|---|---|---|
| `bonded_network` | lateral | 0 / 1 |  | join the points of a chain by springs |
| `thermal_bath` | lateral | 0 / 2 |  | the overdamped thermal bath |
| `membrane_anchor` | lateral | 0 / 2 |  | hold a chain's first domain to its membrane |
| `tip_well` | lateral | 0 / 2 |  | the trans interaction between two EC1 tips |
| `rod_foot` | lateral | 0 / 1 |  | stand a rod on its cell's surface point |
| `rod_reaction` | lateral | 0 / 1 |  | the rods' reaction on their cell points |
| `surface_spacer` | lateral | 0 / 1 |  | hold the facing layers of two cells at a set gap |

### channel_ops

| name | kind | models / impls | aliases | title |
|---|---|---|---|---|
| `pair_potential` | exchange | 6 / 5 |  | apply pair forces between sets |

### contact_ops

| name | kind | models / impls | aliases | title |
|---|---|---|---|---|
| `mesh_contact` | lateral | 3 / 2 |  | push material off a moving surface |
| `mesh_inside` | lateral | 0 / 1 |  | measure matrix inside the surface |
| `bm_sense` | structural | 0 / 2 |  | sense the basement membrane below |
| `plate_confine` | structural | 1 / 1 |  | confine between two plates |
| `seed_plate` | seed | 0 / 1 |  | lay out the plate |
| `surface_drive` | structural | 1 / 1 | `plate_drive` | move the surface at a set rate |

### continuous_engine

| name | kind | models / impls | aliases | title |
|---|---|---|---|---|
| `mpm_emit` | structural | 0 / 1 |  | release new material at the inlet |
| `mpm_drain` | structural | 0 / 1 |  | remove material past the outlet |

### contractile_ops

| name | kind | models / impls | aliases | title |
|---|---|---|---|---|
| `active_strain` | lateral | 2 / 1 |  | contract cells by shortening rest length |
| `material_from_cell` | lateral | 0 / 1 |  | copy cell stiffness to its points |

### diffusion_reaction

| name | kind | models / impls | aliases | title |
|---|---|---|---|---|
| `cell_geometry` | aggregate | 0 / 1 |  | measure cell geometry |
| `cell_neighbours` | rewire | 3 / 1 |  | find neighbouring cells |
| `seed_cell_chem` | seed | 5 / 1 | `cell_chem_seed` | set the initial morphogen |
| `cell_chem_diffuse` | lateral | 3 / 2 |  | diffuse morphogen between neighbours |
| `cell_chem_react` | lateral | 17 / 1 |  | run autocatalytic reactions |
| `cell_grow` | lateral | 9 / 1 |  | grow where morphogen is high |
| `interface_tension` | lateral | 0 / 1 |  | tighten the activator boundary |
| `cell_chem_from_shape` | lateral | 5 / 0 |  | make chemistry from curvature |
| `cell_shape_probe` | lateral | 2 / 0 |  | measure the cell shape index |

### ecm_ops

| name | kind | models / impls | aliases | title |
|---|---|---|---|---|
| `ecm_stress` | lateral | 0 / 1 |  | colour the matrix by compression |

### encoding_ops

| name | kind | models / impls | aliases | title |
|---|---|---|---|---|
| `hash_encoding` | field | 1 / 0 |  | learn a field from coordinates |
| `voxelize` | exchange | 0 / 1 |  | spread values onto a grid |

### field_ops

| name | kind | models / impls | aliases | title |
|---|---|---|---|---|
| `diffuse` | field | 0 / 2 |  | diffuse the field |
| `decay` | field | 0 / 1 |  | evaporate the field |
| `pacemaker` | field | 0 / 1 |  | tick a shared clock |
| `activation_pulse` | field | 0 / 1 |  | paint a timed activation wave |

### flatten_ops

| name | kind | models / impls | aliases | title |
|---|---|---|---|---|
| `cell_normal` | lateral | 0 / 1 |  | fit each cell's layer normal from its neighbours |
| `frame_rotation` | lateral | 2 / 0 |  | turn the whole body by a random rotation before the substep |

### gastrulation_ops

| name | kind | models / impls | aliases | title |
|---|---|---|---|---|
| `cell_populate` | divide | 6 / 0 |  | grow cells by adding points |
| `apical_constriction` | lateral | 0 / 1 |  | constrict the apical face |
| `apical_junction` | lateral | 0 / 1 |  | bond surface cells into a sheet |
| `active_rotation` | lateral | 0 / 1 |  | rotate the vegetal mass |

### interaction_ops

| name | kind | models / impls | aliases | title |
|---|---|---|---|---|
| `attraction_repulsion` | lateral | 0 / 6 |  | attract and repel particles |
| `radius_graph` | rewire | 1 / 2 |  | connect neighbours within a radius |

### junction_ops

| name | kind | models / impls | aliases | title |
|---|---|---|---|---|
| `junction_myosin` | structural | 2 / 1 |  | recruit myosin to stretched junctions |
| `junction_sync` | rewire | 0 / 1 |  | re-index junction myosin after rewiring |
| `junction_pcp` | lateral | 2 / 1 |  | set up planar cell polarity |

### membrane_ops

| name | kind | models / impls | aliases | title |
|---|---|---|---|---|
| `bm_seed` | seed | 0 / 2 |  | lay down the basement membrane |
| `bm_bond` | lateral | 1 / 1 |  | crosslink the basement membrane |
| `bm_unbond` | rewire | 1 / 1 |  | break overstretched crosslinks |
| `bm_remodel` | lateral | 0 / 1 |  | remodel membrane crosslinks |
| `bm_contact` | lateral | 0 / 2 |  | keep the membrane out of the tissue |
| `bm_secrete` | structural | 0 / 3 |  | secrete new membrane as tissue grows |

### motion_ops

| name | kind | models / impls | aliases | title |
|---|---|---|---|---|
| `drag` | lateral | 0 / 1 |  | slow motion by viscous drag |
| `gravity` | lateral | 2 / 1 |  | apply uniform gravity |
| `cohesion` | lateral | 0 / 1 |  | steer toward the neighbours' centre |
| `separation` | lateral | 0 / 1 |  | steer away from close neighbours |
| `velocity_align` | lateral | 0 / 1 | `alignment` | match the neighbours' velocity |

### mpm_ops

| name | kind | models / impls | aliases | title |
|---|---|---|---|---|
| `mpm_scatter` | exchange | 0 / 4 | `p2g` | spread point mass onto the grid |
| `mpm_grid_update` | field | 0 / 7 |  | solve grid velocities |
| `mpm_grid_viscosity` | field | 0 / 1 |  | damp grid shear implicitly |
| `mpm_gather` | exchange | 0 / 4 | `g2p` | move points with the grid |
| `mpm_strain` | lateral | 0 / 4 |  | update material strain |
| `mpm_turgor` | lateral | 1 / 1 | `osmotic_pressure` | inflate cells with turgor pressure |
| `mpm_viscosity` | lateral | 0 / 2 |  | damp shear inside liquid cells |
| `mpm_anchor` | lateral | 1 / 1 |  | anchor a body with springs |
| `active_force` | exchange | 10 / 1 | `pulse_to_contraction` | contract by an active body force |
| `seed_from_segmentation` | seed | 0 / 1 |  | build cells from a segmentation |
| `set_material` | seed | 1 / 1 |  | set what the body is made of |
| `cell_nucleus` | structural | 1 / 0 |  | keep each cell's nucleus the innermost share of its points |
| `ecm_seed` | seed | 0 / 1 | `seed_ecm` | lay out the fibre matrix |
| `mpm_tear` | lateral | 1 / 1 |  | tear material past a breaking stretch |
| `mpm_rigid_body` | exchange | 0 / 2 |  | move a rigid projectile through material |
| `mpm_sponge` | field | 0 / 1 |  | absorb waves at the walls |
| `mpm_grid_contact` | field | 0 / 1 |  | let two materials push without sticking |
| `mpm_grid_noise` | field | 0 / 2 |  | stir the grid with coherent noise |

### mpm_plasticity_ops

| name | kind | models / impls | aliases | title |
|---|---|---|---|---|
| `mpm_plasticity` | lateral | 0 / 1 |  | return the strain to the yield surface |

### organelle_ops

| name | kind | models / impls | aliases | title |
|---|---|---|---|---|
| `organelle_seed` | seed | 0 / 1 |  | place organelles inside cells |
| `organelle_project` | structural | 0 / 1 |  | carry organelles with their cell |
| `organelle_express` | structural | 0 / 1 |  | make and remove organelles |

### particlegraph_ops

| name | kind | models / impls | aliases | title |
|---|---|---|---|---|
| `seed_field` | seed | 5 / 1 |  | set the initial field from an image |
| `react_diffuse` | field | 3 / 1 |  | diffuse a field with a coefficient per region |
| `spring_matrix` | lateral | 0 / 1 |  | tie particles in pairs with random springs |

### protein_ops

| name | kind | models / impls | aliases | title |
|---|---|---|---|---|
| `protein_seed` | seed | 0 / 1 |  | place protein clusters on cells |
| `protein_project` | structural | 0 / 1 |  | keep protein clusters in their region |
| `protein_express` | structural | 0 / 1 |  | make and remove protein clusters |

### relation_ops

| name | kind | models / impls | aliases | title |
|---|---|---|---|---|
| `project` | lateral | 0 / 1 |  | send a rate along connections |
| `readout` | lateral | 0 / 1 |  | send a value along connections |

### rod_ops

| name | kind | models / impls | aliases | title |
|---|---|---|---|---|
| `rod_seed` | seed | 0 / 1 |  | lay out straight rods |
| `rod_elastic` | lateral | 0 / 1 |  | resist rod stretching and bending |
| `rod_base` | lateral | 0 / 1 |  | hold the rod base in place |
| `rod_motor` | lateral | 0 / 1 |  | drive bending along the filament |

### seed_ops

| name | kind | models / impls | aliases | title |
|---|---|---|---|---|
| `seed_positions` | seed | 26 / 1 |  | set initial positions |
| `cloud_seed` | seed | 0 / 1 |  | place a measured point cloud |

### spheroid_ops

| name | kind | models / impls | aliases | title |
|---|---|---|---|---|
| `cell_fate` | structural | 0 / 1 |  | give each cell a fate by its signal level |
| `polarity_guide` | exchange | 0 / 1 |  | turn each cell's planar polarity along the level lines of a signal (round a spot of it) |

### surface_protein_ops

| name | kind | models / impls | aliases | title |
|---|---|---|---|---|
| `bm_clutch` | structural | 0 / 1 |  | bind integrin to laminin |
| `bm_mass` | structural | 0 / 1 |  | turn over basement membrane matter |
| `cell_protein_level` | structural | 0 / 1 |  | relax cell protein levels with memory |

### vertex_ops

| name | kind | models / impls | aliases | title |
|---|---|---|---|---|
| `seed_mesh` | seed | 1 / 4 | `mesh_seed` | build the cell mesh |
| `cell_mechanics` | lateral | 3 / 4 |  | relax cell shapes |
| `cell_divide` | divide | 17 / 4 |  | divide cells above a size |
| `cell_die` | die | 12 / 3 |  | remove marked cells |
| `seed_cycle` | seed | 0 / 1 |  | spread cells over the cell cycle |
| `cell_cycle` | lateral | 6 / 1 |  | advance the cell cycle |
| `topo_record` | structural | 0 / 1 |  | record the mesh each frame |
| `edge_flip` | rewire | 1 / 1 |  | swap neighbours |
