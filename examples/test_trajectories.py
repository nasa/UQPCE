import unittest
import numpy as np
import openmdao.api as om
import dymos as dm
from practice.kinematics.TrajFinalExample import EOM


class TestCost(unittest.TestCase):
    def setUp(self):
        # Drag coefficient originates from uniform distribution sample point
        cd_0 = 0.42364
        # Theta originates from normal distribution sample point, converted to radians
        theta_0 = 30.1797 * (np.pi/180)
        vx = 80 * np.cos(theta_0)
        vy = 80 * np.sin(theta_0)
        # Area calculated using mass = 12
        A = 0.014289

        prob = om.Problem()
        vals = {
            'c_d': cd_0,
            'A': A,
            'g': 9.80665,
            'rho_0': 1.22,
            'm': 12
        }

        traj = dm.Trajectory()
        tx = dm.PicardShooting(num_segments=1, nodes_per_seg=11, solve_segments='forward')
        tx2 = dm.PicardShooting(num_segments=1, nodes_per_seg=11, solve_segments='forward')
    
        ascent = traj.add_phase('ascent', dm.Phase(ode_class=EOM, transcription=tx))

        ascent.set_time_options(fix_initial=True, fix_duration=True)

        ascent.add_state('x', rate_source='x_dot', fix_initial=True, fix_final=False)
        ascent.add_state('y', rate_source='y_dot', fix_initial=True, fix_final=False)
        ascent.add_state('vx', rate_source='vx_dot', input_initial=True, fix_initial=False, fix_final=False)
        ascent.add_state('vy', rate_source='vy_dot', input_initial=True, fix_initial=False, fix_final=False)

        ascent.add_parameter('c_d', units='unitless', val=vals['c_d'])
        ascent.add_parameter('A', units='m**2', val=vals['A'])
        ascent.add_parameter('g', units='m/s**2', val=vals['g'])
        ascent.add_parameter('rho_0', units='kg/m**3', val=vals['rho_0'])
        ascent.add_parameter('m', units='kg', val=vals['m'])

        ascent.add_boundary_balance(param='t_duration', name='vy', tgt_val=0.0, loc='final', lower=0.1, upper=100.0)
        ascent.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=100, iprint=0, debug_print=True)
        ascent.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
        ascent.nonlinear_solver.debug_print = True
        ascent.linear_solver = om.DirectSolver()

        descent = traj.add_phase('descent', dm.Phase(ode_class=EOM, transcription=tx2))

        descent.set_time_options(input_initial=True, fix_duration=True)

        descent.add_state('x', rate_source='x_dot', input_initial=True, fix_final=False)
        descent.add_state('y', rate_source='y_dot', input_initial=True, fix_final=False)
        descent.add_state('vx', rate_source='vx_dot', input_initial=True, fix_final=False)
        descent.add_state('vy', rate_source='vy_dot', input_initial=True, fix_final=False)

        descent.add_parameter('c_d', units='unitless', val=vals['c_d'])
        descent.add_parameter('A', units='m**2', val=vals['A'])
        descent.add_parameter('g', units='m/s**2', val=vals['g'])
        descent.add_parameter('rho_0', units='kg/m**3', val=vals['rho_0'])
        descent.add_parameter('m', units='kg', val=vals['m'])

        descent.add_boundary_balance(param='t_duration', name='y', tgt_val=0.0, loc='final', lower=0.1, upper=100.0)
        descent.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=100, iprint=0, debug_print=True)
        descent.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
        descent.nonlinear_solver.debug_print = True
        descent.linear_solver = om.DirectSolver()

        traj.link_phases(phases=('ascent', 'descent'), connected=True, vars='*')

        prob.model.add_subsystem('traj', traj)
        
        prob.setup()

        ranges = {
            't_init': 0,
            't_dur': 5,
            'x_init': 0,
            'x_dur': 100,
            'y_init': 0,
            'y_dur': 50
        }
        
        ascent.set_time_val(initial=ranges['t_init'], duration=ranges['t_dur'])
        ascent.set_state_val('x', [ranges['x_init'], ranges['x_dur']])
        ascent.set_state_val('y', [ranges['y_init'], ranges['y_dur']])
        ascent.set_state_val('vx', [vx, vx])
        ascent.set_state_val('vy', [vy, 0])

        descent.set_time_val(initial=ranges['t_init'], duration=ranges['t_dur'])
        descent.set_state_val('x', [ranges['x_init'], ranges['x_dur']])
        descent.set_state_val('y', [ranges['y_init'], ranges['y_dur']])
        descent.set_state_val('vx', [vx, vx])
        descent.set_state_val('vy', [0, -vy])

        prob.run_model()
        self.partials = prob.check_partials(out_stream=None, method='fd')
        self.prob = prob
 
    def test_compute(self):
        # Height at max ascent
        y_max = self.prob.get_val('traj.ascent.timeseries.y')[10]
        # Final x-distance
        x_max = self.prob.get_val('traj.descent.timeseries.x')[10]

        # True value computations
        y_max_true = 77.6669
        x_max_true = 506.4799

        self.assertTrue(
            np.isclose(y_max, y_max_true), msg="Max height error."
        )
        self.assertTrue(
            np.isclose(x_max, x_max_true), msg="Max distance error."
        )


if __name__ == '__main__':
    suite = unittest.TestSuite()
    unittest.main()