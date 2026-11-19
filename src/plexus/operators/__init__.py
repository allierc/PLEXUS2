"""The operator library. Importing this package self-registers every operator.

The modules are grouped BY MECHANISM, not one file per operator. The alternative is unreadable in
a specific way rather than merely long: finding out what acts on an MPM particle would mean
opening eight files, and the eight could never be compared because they would never be on the
screen together. The two implementations of `cell_mechanics` -- the 3D vertex model and the
monolayer -- settle it. They are one contract with two bodies, and reading them one after the
other is how anyone can tell which one a specification is getting.

    vertex_ops           the 3D vertex model: seed, geometry, mechanics, grow, divide, die, T1
    diffusion_reaction   chemistry on the cell GRAPH, and the two shape-chemistry couplings
    junction_ops         myosin, on junctions and across the apex, plus the cytokinetic ring
    ecm_ops              the forces the matrix carries, and the stiff blocks that confine it
    membrane_ops         the basement membrane, its crosslink network, and the integrin links
    surface_protein_ops  the budding proteins of exp 11 Phase 3 on a live membrane: integrin-laminin bonds
    contact_ops          where a triangulated surface meets a continuum, both directions
    cell_ops             the cell as a composition: compartment pieces, the atlas that places
                         them, and the centroid that aggregates a level onto the one above
    mpm_ops              MLS-MPM: the grid, the four-step cycle, and the forces on it
    motion_ops           single-body motion: drag, glide, sediment, walls, gravity
    encoding_ops         fields that REPRESENT rather than simulate: hash_encoding, voxelize
    interaction_ops      pairwise laws, and the relation they act over
    field_ops            a continuum bound to a set: deposit / diffuse / decay / sense
    relation_ops         what travels along a relation between two kinds: `project`, the
                         general gather/weight/scatter

Every module opens with its own contracts listed IN THE ORDER THEY APPEAR IN THE FILE, then the
models and implementations of those contracts. Each contract's docstring states the mechanism, its
typed morphism, the equation it implements, every symbol in that equation with its units, and the
publication it comes from where there is one.

There are no per-operator module names. A module that re-exports one operator reads as a separate
mechanism when it is one line of a file, so an import of a name that used to exist raises rather
than resolving -- which is the correct answer and not a regression.
"""
from __future__ import annotations

# --- the mechanism modules -------------------------------------------------------------------
from . import encoding_ops          # noqa: F401  hash_encoding -- a learnable field from its
                                   #               own coordinates; voxelize -- a set, splatted
from . import interaction_ops       # noqa: F401  radius_graph, attraction_repulsion, squared_law,
#                                                 stillinger_weber
from . import motion_ops            # noqa: F401  drag, glide, velocity_cruise, sediment,
#                                                 attractor_flow, gravity, bounce,
#                                                 cohesion, separation, velocity_align (the boids rules)
from . import field_ops             # noqa: F401  the grid field, deposit, diffuse, decay, sense,
#                                                 chemotax, playback, pacemaker, activation_pulse, signal
from . import mpm_ops               # noqa: F401  mpm_grid + p2g/grid_update/g2p/strain, seed_ecm,
#                                                 anchor, spin, apply_material_map, fenced oracle
from . import vertex_ops            # noqa: F401  seed_mesh, cell_mechanics, cell_divide, cell_die,
#                                                 edge_flip, topo_record
from . import seed_ops              # noqa: F401  seed_positions -- where a set's entities start
from . import diffusion_reaction    # noqa: F401  seed_cell_chem, cell_chem_diffuse/react,
from . import organelle_ops         # noqa: F401  organelle_seed / _project -- pieces with a radius inside a cell
from . import protein_ops           # noqa: F401  protein_seed / _project / _express -- receptor and
                                    #              motor clusters as point entities on a cap of the tissue
#                                                 cell_geometry, cell_grow, cell_chem_from_shape,
#                                                 cell_shape_probe, interface_tension/push
from . import junction_ops          # noqa: F401  junction_myosin (default|two_pool), junction_sync,
#                                                 medioapical_myosin, cytokinetic_ring, junction_pcp
from . import ecm_ops               # noqa: F401  ecm_stress/from_cell, cell_exclude,
#                                                 block_seed/stress
from . import membrane_ops          # noqa: F401  bm_*, adhesion_*, integrin_*
from . import surface_protein_ops   # noqa: F401  bm_clutch -- integrin beta1-laminin bonds on the live membrane
from . import contact_ops           # noqa: F401  mesh_contact, mesh_inside, surface_track,
#                                                 plate_confine, bm_sense, ecm_load, ecm_gate_growth
from . import cell_ops             # noqa: F401  compartment (entity), seed_cell_atlas,
#                                                 aggregate_centroid, and the
#                                                 motility pair: substrate_traction, protrusion
from . import contractile_ops     # noqa: F401  active_strain, material_from_cell -- a muscle's rest length, per cell
from . import relation_ops          # noqa: F401  project -- one set's state across a relation
#                                                 onto another set's block (W_in, W_out, and
#                                                 circuit -> circuit are all this one operator)
# `cilia_ops` WAS HERE AND IS DELETED. It held the MPM cilium -- cilium_seed, cilium_pose_map,
# cilium_kinematics, cilium_torque, cilium_anchor and the seeds that placed them -- and the whole
# construction has a floor it cannot clear: MLS-MPM carries one velocity per grid node, so a body
# thinner than a grid cell is advected rather than swum, and a Platynereis cilium is 0.25 um
# against a 4.08 um cell. Every attempt widened the cilium until the grid could hold it and none
# of them was a cilium. `rod_ops` below is the replacement and not a refinement: a filament whose
# thickness is a drag coefficient has no such floor. Its one operator that was never ciliary,
# polar_active_stress[driven], moved to `cell_ops` beside the class it subclasses.
from . import rod_ops               # noqa: F401  a cilium as a DISCRETE ELASTIC ROD: rod_seed,
#                                                 rod_elastic (stretch + bend, momentum-exact) and
#                                                 rod_base (the base held in place, held in
#                                                 direction, and driven -- one moment, three laws).
#                                                 a rod has no such floor, so this is how the model
#                                                 gets a cilium that is 0.25 um thick.
from . import channel_ops           # noqa: F401  membrane pores and channels: elastic_network, pair_potential,
#                                                 brownian, tether, radial_drive (the patch frame),
#                                                 electrolyte_conduction (the current through the open lumen),
#                                                 pore_probe
from . import cleavage_ops         # noqa: F401  seed_positions[egg], cell_divide[cleave], mpm_grid_update[body_csf]
#                                                 -- a frog egg cleaving, experiments/exp23_frog_cleavage
from . import gastrulation_ops     # noqa: F401  seed_positions[blastula] -- a frog blastula gastrulating, experiments/exp25_frog_gastrulation
from . import adhesion_ops         # noqa: F401  seed_positions[cell_grid], junction_contact/kinetics, cell_adhesion, cell_pull -- experiments/exp27_cell_adhesion
from . import bond_population_ops  # noqa: F401  bond_binding[population], bond_springs[population] -- the junction one level down (exp29, 2026-11-02)
from . import chain_ops              # noqa: F401  bonded_network, thermal_bath, membrane_anchor, tip_well, seed_positions[chains] -- the cadherin chain, exp37 (2026-11-02)
from . import flatten_ops          # noqa: F401  deform_control[cell_flatten] -- each cell's own volume-preserving flattening rate, exp29 minimal 3 x 3 rig
from . import bm_shell_ops          # noqa: F401  a basement membrane shell around MPM cells: seed_positions[bm_sphere], bm_shell, bm_cell_contact, bm_cell_apply -- exp27
# The continuous-flow operators live with their engine but must still register here. `mpm_emit`
# and `mpm_drain` are defined in plexus/continuous_engine.py, beside the run() that refuses a
# specification using one without the other. Importing them only when that engine is selected
# would be too late: `schema.load` validates every operator NAME before anything chooses an
# engine, so a specification naming mpm_emit would fail before the engine key was read. That
# module imports models.base and models.registry only, so there is no cycle back into this one.
from plexus import continuous_engine   # noqa: F401  mpm_emit / mpm_drain

__all__ = ["encoding_ops", "interaction_ops", "motion_ops", "field_ops", "mpm_ops",
           "vertex_ops", "diffusion_reaction", "junction_ops", "ecm_ops", "membrane_ops",
           "contact_ops", "cell_ops", "relation_ops", "rod_ops", "continuous_engine"]
from . import mpm_warp_pp_ops          # noqa: F401  exp27 speed audit: test 5a/5b: mpm_scatter[warp_pp] and mpm_gather[warp_pp], the Warp scatter and gather on the per-cell grid blocks
from . import mpm_warp_pp_grid_ops          # noqa: F401  exp27 speed audit: test 5c: mpm_grid_update[warp_pp_balanced], body_csf_balanced as Warp kernels over the per-cell grid blocks
from . import bm_shell_warp_ops          # noqa: F401  exp27 speed audit: test 5d: bm_cell_contact[warp], the membrane-to-cell contact as one Warp kernel on a hash grid of the nodes
from . import adhesion_warp_ops          # noqa: F401  exp27 speed audit: test 5e: junction_contact[hashgrid] and active_force[sphere_crawl, hashgrid], the kd-tree searches as Warp hash-grid queries on the GPU
from . import contact_warp_pp_ops       # noqa: F401  exp09 big scenes (2026-11-17): pair_potential[cell_contact_warp_pp], the overlap penalty on each cell's own block of lattice nodes, Warp
from . import binding_delta_g_ops       # noqa: F401  exp09 (2026-11-19): bond_binding[delta_g], the unbinding rate of a junction from its pair's binding free energy
from . import substep_warp_ops         # noqa: F401  exp09 speed (2026-11-19): the substep block of a crowd of small MPM bodies as Warp kernels one CUDA graph can replay (thermal_bath[langevin_warp], active_force[box_slab_warp], pair_potential[cell_contact_warp_pp_fixed], bond_springs[warp], apical_junction[warp], mpm_scatter[warp_pp_lean], mpm_grid_update[solved_in_gather], mpm_gather[warp_pp_solve]): 13 x faster on rna1_3d (1.186 -> 0.089 s per frame on one H200)
from . import mpm_warp_polar_ops          # noqa: F401  exp27 speed audit: test 1, mpm_strain[polar] (Maxwell relaxation by the polar factor, one Warp kernel)
from . import mpm_plasticity_ops         # noqa: F401  exp01 demo: mpm_plasticity[law: metal | sand], PhysGaussian's von Mises and Drucker-Prager return maps
from . import mpm_rigid_ops              # noqa: F401  exp01 demo: mpm_rigid_body[implementation: bounce], the rigid projectile with a floor
from . import particlegraph_ops          # noqa: F401  exp01 demo: react_diffuse[heat | fitzhugh_nagumo | rock_paper_scissors], seed_field, spring_matrix (ParticleGraph's mesh models and springs)
from . import spheroid_ops               # noqa: F401  exp05: cell_divide[point_adhesion], point cells that grow and divide only while they touch enough others
