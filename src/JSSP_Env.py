"""
Job Shop Scheduling Problem (JSSP) Environment with Heterogeneous Graph Representation.

This environment represents JSSP as a disjunctive graph using PyTorch Geometric's HeteroData.
State: Heterogeneous graph with two edge types:
    - ('op', 'precedes', 'op'): Conjunctive edges (job flow constraints)
    - ('op', 'competes', 'op'): Disjunctive edges (machine sharing constraints)
"""

import gymnasium as gym
import numpy as np
import torch
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Operation:
    """Represents a single operation in JSSP."""
    job_id: int
    op_id: int  # Operation index within the job
    machine_id: int
    processing_time: float
    start_time: float = -1.0
    completion_time: float = -1.0
    scheduled: bool = False


class JSSPEnv(gym.Env):
    """
    Job Shop Scheduling Problem Environment with Heterogeneous Graph State.

    The state is represented as a disjunctive graph:
    - Nodes: Operations with features [processing_time, completion_percentage, scheduled_flag]
    - Edges:
        * 'precedes': Conjunctive constraints (operation order within jobs)
        * 'competes': Disjunctive constraints (operations sharing same machine)

    Action: Select an eligible operation to schedule next.
    Reward: Negative makespan improvement (sparse) or dense shaping reward.
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        num_jobs: int,
        num_machines: int,
        processing_times: Optional[np.ndarray] = None,
        machine_assignment: Optional[np.ndarray] = None,
        reward_type: str = 'dense',  # 'sparse' or 'dense'
        normalize_features: bool = True,
    ):
        """
        Initialize JSSP Environment.

        Args:
            num_jobs: Number of jobs
            num_machines: Number of machines
            processing_times: Shape (num_jobs, num_machines) - processing time for each operation
            machine_assignment: Shape (num_jobs, num_machines) - machine order for each job
            reward_type: 'sparse' (only at episode end) or 'dense' (shaped reward)
            normalize_features: Whether to normalize node features
        """
        super().__init__()

        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.num_operations = num_jobs * num_machines
        self.reward_type = reward_type
        self.normalize_features = normalize_features

        # Generate random instance if not provided
        if processing_times is None:
            self.processing_times = np.random.randint(1, 100, size=(num_jobs, num_machines))
        else:
            self.processing_times = processing_times

        if machine_assignment is None:
            # Each job visits all machines in random order
            self.machine_assignment = np.array([
                np.random.permutation(num_machines) for _ in range(num_jobs)
            ])
        else:
            self.machine_assignment = machine_assignment

        # Action space: select one operation to schedule
        self.action_space = gym.spaces.Discrete(self.num_operations)

        # Observation space: HeteroData (defined dynamically)
        self.observation_space = gym.spaces.Graph(
            node_space=gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32),
            edge_space=gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        )

        # State variables
        self.operations: List[Operation] = []
        self.machine_queues: Dict[int, List[int]] = {}  # machine_id -> list of op indices
        self.job_progress: np.ndarray = np.zeros(num_jobs, dtype=int)  # Next op to schedule per job
        self.machine_available_time: np.ndarray = np.zeros(num_machines)
        self.current_time: float = 0.0
        self.makespan: float = 0.0
        self.scheduled_count: int = 0

        # For reward shaping
        self.prev_lower_bound: float = 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[HeteroData, dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Initialize operations
        self.operations = []
        op_idx = 0
        for j in range(self.num_jobs):
            for m_idx in range(self.num_machines):
                machine = self.machine_assignment[j, m_idx]
                proc_time = self.processing_times[j, m_idx]
                op = Operation(
                    job_id=j,
                    op_id=m_idx,
                    machine_id=int(machine),
                    processing_time=float(proc_time)
                )
                self.operations.append(op)
                op_idx += 1

        # Initialize machine queues
        self.machine_queues = {m: [] for m in range(self.num_machines)}
        for idx, op in enumerate(self.operations):
            self.machine_queues[op.machine_id].append(idx)

        # Reset state variables
        self.job_progress = np.zeros(self.num_jobs, dtype=int)
        self.machine_available_time = np.zeros(self.num_machines)
        self.current_time = 0.0
        self.makespan = 0.0
        self.scheduled_count = 0

        # Compute initial lower bound
        self.prev_lower_bound = self._compute_lower_bound()

        state = self._get_state()
        info = {'makespan': 0.0, 'scheduled': 0}

        return state, info

    def step(self, action: int) -> Tuple[HeteroData, float, bool, bool, dict]:
        """
        Execute one scheduling step.

        Args:
            action: Index of operation to schedule

        Returns:
            state: HeteroData representing current state
            reward: Scalar reward
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Validate action
        if not self._is_action_valid(action):
            # Invalid action: large penalty
            reward = -1000.0
            terminated = False
            truncated = False
            info = {'makespan': self.makespan, 'scheduled': self.scheduled_count, 'invalid_action': True}
            return self._get_state(), reward, terminated, truncated, info

        # Schedule the operation
        op = self.operations[action]

        # Determine start time
        # Must wait for: (1) previous operation in job, (2) machine availability
        job_ready_time = 0.0
        if op.op_id > 0:
            # Find previous operation in the same job
            prev_op_idx = action - 1
            prev_op = self.operations[prev_op_idx]
            job_ready_time = prev_op.completion_time

        machine_ready_time = self.machine_available_time[op.machine_id]
        start_time = max(job_ready_time, machine_ready_time)

        # Update operation
        op.start_time = start_time
        op.completion_time = start_time + op.processing_time
        op.scheduled = True

        # Update machine availability
        self.machine_available_time[op.machine_id] = op.completion_time

        # Update job progress
        self.job_progress[op.job_id] += 1
        self.scheduled_count += 1

        # Update current time and makespan
        self.current_time = max(self.current_time, op.completion_time)
        self.makespan = max(self.makespan, op.completion_time)

        # Compute reward
        reward = self._compute_reward()

        # Check if done
        terminated = (self.scheduled_count == self.num_operations)
        truncated = False

        info = {
            'makespan': self.makespan,
            'scheduled': self.scheduled_count,
            'action_op': f"J{op.job_id}_M{op.machine_id}",
            'invalid_action': False
        }

        return self._get_state(), reward, terminated, truncated, info

    def _get_state(self) -> HeteroData:
        """
        Construct heterogeneous graph state representation.

        Returns:
            HeteroData with:
                - Node features: [processing_time, completion_percentage, scheduled_flag]
                - Edge types: 'precedes' (conjunctive), 'competes' (disjunctive)
        """
        data = HeteroData()

        # Node features
        node_features = []
        for op in self.operations:
            proc_time = op.processing_time
            completion_pct = op.completion_time / max(self.current_time, 1.0) if op.scheduled else 0.0
            scheduled_flag = 1.0 if op.scheduled else 0.0

            features = [proc_time, completion_pct, scheduled_flag]
            node_features.append(features)

        node_features = np.array(node_features, dtype=np.float32)

        # Normalize features
        if self.normalize_features:
            # Normalize processing time
            max_proc_time = np.max(self.processing_times)
            node_features[:, 0] = node_features[:, 0] / max_proc_time
            # completion_pct and scheduled_flag are already in [0, 1]

        data['op'].x = torch.from_numpy(node_features)

        # Conjunctive edges (precedence within jobs)
        precedes_edges = []
        for j in range(self.num_jobs):
            for m_idx in range(self.num_machines - 1):
                src = j * self.num_machines + m_idx
                dst = j * self.num_machines + m_idx + 1
                precedes_edges.append([src, dst])

        if precedes_edges:
            precedes_edges = torch.tensor(precedes_edges, dtype=torch.long).t().contiguous()
            data['op', 'precedes', 'op'].edge_index = precedes_edges
        else:
            data['op', 'precedes', 'op'].edge_index = torch.empty((2, 0), dtype=torch.long)

        # Disjunctive edges (competition for machines)
        competes_edges = []
        for machine_id, op_indices in self.machine_queues.items():
            # Fully connect all operations on the same machine
            for i, src in enumerate(op_indices):
                for dst in op_indices[i+1:]:
                    competes_edges.append([src, dst])
                    competes_edges.append([dst, src])  # Bidirectional

        if competes_edges:
            competes_edges = torch.tensor(competes_edges, dtype=torch.long).t().contiguous()
            data['op', 'competes', 'op'].edge_index = competes_edges
        else:
            data['op', 'competes', 'op'].edge_index = torch.empty((2, 0), dtype=torch.long)

        # Add action mask as node attribute
        action_mask = self._get_action_mask()
        data['op'].action_mask = torch.from_numpy(action_mask).bool()

        return data

    def _get_action_mask(self) -> np.ndarray:
        """
        Get binary mask of valid actions.

        An operation is valid if:
        1. It's not scheduled yet
        2. It's the next operation in its job sequence
        """
        mask = np.zeros(self.num_operations, dtype=np.float32)

        for j in range(self.num_jobs):
            next_op_id = self.job_progress[j]
            if next_op_id < self.num_machines:
                op_idx = j * self.num_machines + next_op_id
                mask[op_idx] = 1.0

        return mask

    def _is_action_valid(self, action: int) -> bool:
        """Check if action is valid."""
        mask = self._get_action_mask()
        return mask[action] > 0

    def _compute_reward(self) -> float:
        """
        Compute reward for the current step.

        Dense reward: Negative of lower bound improvement
        Sparse reward: Only at episode end
        """
        if self.reward_type == 'sparse':
            # Only give reward at episode end
            if self.scheduled_count == self.num_operations:
                return -self.makespan
            else:
                return 0.0
        else:  # dense
            # Shaped reward based on lower bound improvement
            current_lb = self._compute_lower_bound()
            improvement = self.prev_lower_bound - current_lb
            self.prev_lower_bound = current_lb

            # Small penalty for each step + reward for improvement
            return improvement - 0.1

    def _compute_lower_bound(self) -> float:
        """
        Compute a lower bound on the remaining makespan.

        Lower bound = max(
            max machine workload remaining,
            max job workload remaining
        )
        """
        # Machine workload
        machine_workload = self.machine_available_time.copy()

        # Job workload
        job_workload = np.zeros(self.num_jobs)
        for j in range(self.num_jobs):
            remaining_ops = range(self.job_progress[j], self.num_machines)
            for m_idx in remaining_ops:
                op_idx = j * self.num_machines + m_idx
                job_workload[j] += self.operations[op_idx].processing_time

        # Add job completion times for jobs that started
        for j in range(self.num_jobs):
            if self.job_progress[j] > 0:
                last_scheduled_op_idx = j * self.num_machines + self.job_progress[j] - 1
                job_workload[j] += self.operations[last_scheduled_op_idx].completion_time

        return max(np.max(machine_workload), np.max(job_workload))

    def render(self, mode='human'):
        """Render the current state."""
        if mode == 'human':
            print(f"\n{'='*60}")
            print(f"JSSP State: {self.scheduled_count}/{self.num_operations} operations scheduled")
            print(f"Current Makespan: {self.makespan:.2f}")
            print(f"{'='*60}")

            # Print Gantt chart representation
            for j in range(self.num_jobs):
                print(f"\nJob {j}:")
                for m_idx in range(self.num_machines):
                    op_idx = j * self.num_machines + m_idx
                    op = self.operations[op_idx]
                    status = "DONE" if op.scheduled else "PENDING"
                    if op.scheduled:
                        print(f"  Op{m_idx} [M{op.machine_id}]: {op.start_time:.1f} -> {op.completion_time:.1f} ({status})")
                    else:
                        print(f"  Op{m_idx} [M{op.machine_id}]: -- -> -- ({status})")

            print(f"\nMachine Availability:")
            for m in range(self.num_machines):
                print(f"  Machine {m}: {self.machine_available_time[m]:.2f}")

    def get_optimal_makespan(self) -> Optional[float]:
        """
        Return known optimal makespan if available (for benchmark instances).
        Override this method for specific benchmark problems.
        """
        return None
