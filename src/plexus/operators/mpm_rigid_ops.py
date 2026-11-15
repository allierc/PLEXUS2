"""exp01 twin of the PhysGaussian step (2026-11-09, Cedric: "the white balls bouncing instead of going through the bottom plane"):
`mpm_rigid_body[implementation: bounce]`, the rigid projectile with a floor.

`mpm_rigid_body` has no floor. It stops exchanging momentum with the grid once the body is within its capsule radius + 2 grid cells of any wall of the
box, but keeps applying gravity, so a ball that has crushed the material under it leaves the box through the bottom plane. This variant adds the one
missing contact: a horizontal plane at height `floor` that the lowest point of the body cannot pass, and which returns the downward speed it removes
as an upward speed reduced by `restitution`.
"""
from __future__ import annotations

import torch

from plexus.models.registry import register_operator
from plexus.operators.mpm_ops import MPMRigidBody


@register_operator("mpm_rigid_body", implementation="bounce", family="mpm", set="particle", kind="exchange",
                   title="move a rigid projectile through material, bouncing on a floor",
                   equation=r"""$$y_{\min}=y-\left(h\,|\hat a_y|+r\right)<y_f,\ V_y<0:\qquad y \leftarrow y+(y_f-y_{\min}),\quad V_y \leftarrow -e\,V_y$$""")
class MPMRigidBodyBounce(MPMRigidBody):
    r"""mpm_rigid_body with a floor: the body's lowest point cannot pass the plane y = `floor`, and the speed it had into the plane comes back as `restitution` times that speed, upward.
    particle set (the body's own points) <-> mpm_grid: as mpm_rigid_body, then lifts the body onto the plane and reverses its vertical velocity (a rigid translation; the points' `pos` and `vel` are rewritten).

        y_{min}=y-\left(h\,|\hat a_y|+r\right)<y_f,\ V_y<0:\qquad y \leftarrow y+(y_f-y_{min}),\quad V_y \leftarrow -e\,V_y
        y              height of the body's centre
        h              half-length of the capsule along its axis (0 for a sphere)
        \hat a_y       vertical component of the capsule's unit axis
        r              capsule radius
        y_f            `floor`, height of the plane, world length
        V_y            vertical velocity of the body
        e              `restitution`, the fraction of the downward speed returned upward (0 sticks to the plane, 1 returns all of it)

    The plane is the bottom face of the box when `floor` is 0, the grid's own wall; the body's exchange with the grid is still switched off within r + 2 cells of that wall (the parent's rule), so the last 2 cells of the fall are free flight.

    Reference: Stronge, W. J. (2000). Impact Mechanics. Cambridge University Press (the coefficient of restitution as the ratio of rebound to approach speed); the rigid-body coupling is mpm_rigid_body's."""
    EXPLAIN = (
        r"A rigid body moving through the material, with a floor. Everything is as in `mpm_rigid_body`; in addition the lowest point of the body "
        r"cannot pass the plane $y=y_f$: when it would, the body is lifted back onto the plane and its vertical velocity $V_y$ is reversed and "
        r"multiplied by the restitution $e$ ($e=0$ stops the body on the plane, $e=1$ returns all of its downward speed)."
    )
    PARAM_UNITS = {**MPMRigidBody.PARAM_UNITS, 'floor': 'length', 'restitution': 'dimensionless'}
    SYMBOLS = {**MPMRigidBody.SYMBOLS, 'y': "height of the body's centre", 'h': 'half-length of the capsule along its axis (0 for a sphere)',
               '\\hat a_y': "vertical component of the capsule's unit axis", 'r': 'capsule radius', 'y_f': '`floor`, height of the plane, world length',
               'V_y': 'vertical velocity of the body', 'e': '`restitution`, the fraction of the downward speed returned upward'}
    REFERENCE = MPMRigidBody.REFERENCE + " Stronge, W. J. (2000). Impact Mechanics. Cambridge University Press."
    MECHANISM_TAGS = MPMRigidBody.MECHANISM_TAGS + ["floor", "restitution"]
    PARAM_ROLES = {**MPMRigidBody.PARAM_ROLES, "floor": "height_of_the_plane_the_body_bounces_on",
                   "restitution": "fraction_of_the_downward_speed_returned_upward"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.floor = float(params.get("floor", 0.0))
        self.restitution = float(params["restitution"])
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError(f"mpm_rigid_body[bounce]: restitution must lie in [0, 1], got {self.restitution}")

    def forward(self, H, mask=None):
        out = super().forward(H, mask)
        p = H.level(self.at)
        X, V = p.rb_x, p.rb_v
        if getattr(self, "_reach", None) is None:
            self._reach = self._h * abs(float(self._ax[1])) + self._r
        y_min = X[1:2] - self._reach
        hit = (y_min < self.floor) & (V[1:2] < 0)
        X[1:2].add_(torch.where(hit, self.floor - y_min, torch.zeros_like(y_min)))
        V[1:2].copy_(torch.where(hit, -self.restitution * V[1:2], V[1:2]))
        a, b = p.state_schema["pos"]
        p.state[:, a:b].copy_((X[None, :] + p.rb_off).to(p.state.dtype))
        if "vel" in p.state_schema:
            v0, v1 = p.state_schema["vel"]
            p.state[:, v0:v1].copy_(V[None, :].expand(p.state.shape[0], 3).to(p.state.dtype))
        return out
