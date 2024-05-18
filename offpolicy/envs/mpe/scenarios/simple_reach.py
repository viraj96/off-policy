import numpy as np
import networkx as nx
from offpolicy.envs.mpe.scenario import BaseScenario
from offpolicy.envs.mpe.core import World, Agent, Landmark

from pud.envs.simple_navigation_env import WALLS, resize_walls, thin_walls


class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        world.world_length = args.episode_length

        # Set any world properties first
        world.dim_c = 2
        world.collaborative = True  # type: ignore
        world.num_agents = args.num_agents
        world.num_landmarks = args.num_landmarks

        self.discrete_action = False

        # Add agents
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.silent = True
            agent.collide = True
            agent.name = "agent %d" % i

        # Add landmarks
        world.landmarks = [Landmark() for i in range(world.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.collide = False
            landmark.movable = False
            landmark.name = "landmark %d" % i

        if args.thin and args.resize_factor > 1:
            world.maze_walls = thin_walls(WALLS[args.walls], args.resize_factor)  # type: ignore
        elif not args.thin and args.resize_factor > 1:
            world.maze_walls = resize_walls(WALLS[args.walls], args.resize_factor)  # type: ignore
        else:
            world.maze_walls = WALLS[args.walls]  # type: ignore

        (self.height, self.width) = world.maze_walls.shape  # type: ignore
        # for i in range(world.maze_walls.shape[0]):  # type: ignore
        #     for j in range(world.maze_walls.shape[1]):  # type: ignore
        #         if world.maze_walls[i, j] == 1:  # type: ignore
        #             world.landmarks.append(Landmark())
        #             world.landmarks[-1].collide = True
        #             world.landmarks[-1].movable = False
        #             world.landmarks[-1].state.p_pos = np.array(self.normalize_obs([i, j]), dtype=np.float32)  # type: ignore
        #             world.landmarks[-1].state.p_vel = np.zeros(world.dim_p)  # type: ignore
        #             world.landmarks[-1].state.c = np.zeros(world.dim_c)  # type: ignore
        #             world.landmarks[-1].name = "wall %d %d" % (i, j)

        self.apsp = self.compute_apsp(world.maze_walls)  # type: ignore

        self._min_dist = args.min_dist
        self._max_dist = args.max_dist
        self._difficulty = args.difficulty
        self._prob_constraint = args.prob_constraint
        self._threshold_distance = args.threshold_distance

        # Make initial conditions
        self.reset_world(world)
        return world

    def compute_apsp(self, walls):
        (height, width) = walls.shape
        g = nx.Graph()
        # Add all the nodes
        for i in range(height):
            for j in range(width):
                if walls[i, j] == 0:
                    g.add_node((i, j))

        # Add all the edges
        for i in range(height):
            for j in range(width):
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == dj == 0:
                            continue  # Don't add self loops
                        if i + di < 0 or i + di > height - 1:
                            continue  # No cell here
                        if j + dj < 0 or j + dj > width - 1:
                            continue  # No cell here
                        if walls[i, j] == 1:
                            continue  # Don't add edges to walls
                        if walls[i + di, j + dj] == 1:
                            continue  # Don't add edges to walls
                        g.add_edge((i, j), (i + di, j + dj))

        # dist[i, j, k, l] is path from (i, j) -> (k, l)
        dist = np.full((height, width, height, width), np.float32("inf"))
        for (i1, j1), dist_dict in nx.shortest_path_length(g):
            for (i2, j2), d in dist_dict.items():
                dist[i1, j1, i2, j2] = d
        return dist

    def discretize_state(self, state, resolution=1.0):
        (i, j) = np.floor(resolution * state).astype(np.int64)
        # Round down to the nearest cell if at the boundary.
        if i == self.height:
            i -= 1
        if j == self.width:
            j -= 1
        return (i, j)

    def get_distance(self, state1, state2):
        (i1, j1) = self.discretize_state(state1)
        (i2, j2) = self.discretize_state(state2)
        return self.apsp[i1, j1, i2, j2]

    def is_blocked(self, state, world):
        # Check if state is outside the world
        if state[0] < 0 or state[0] >= self.height:
            return True
        if state[1] < 0 or state[1] >= self.width:
            return True
        (i, j) = self.discretize_state(state)
        return world.maze_walls[i, j] == 1

    def normalize_obs(self, obs):
        return np.array([obs[0] / float(self.height), obs[1] / float(self.width)])

    def set_sample_goal_args(self, min_dist, max_dist, prob_constraint):
        self._min_dist = min_dist
        self._max_dist = max_dist
        self._prob_constraint = prob_constraint

    def set_difficulty(self, difficulty):
        self._difficulty = difficulty
        self.set_sample_goal_args(prob_constraint=1,
                                  min_dist=max(0, self.max_goal_dist * (difficulty - 0.05)),
                                  max_dist=self.max_goal_dist * (difficulty + 0.05))

    def sample_empty_state(self, world):
        candidate_states = np.where(world.maze_walls == 0)
        num_candidate_states = len(candidate_states[0])
        state_index = np.random.choice(num_candidate_states)
        state = np.array(
            [candidate_states[0][state_index], candidate_states[1][state_index]],
            dtype=np.float32,
        )
        state += np.random.uniform(size=2)
        assert not self.is_blocked(state, world)

        return self.normalize_obs(state)

    def sample_goal_unconstrained(self, agent, world):
        return self.sample_empty_state(world)

    def sample_goal_constrained(self, agent, world, min_dist, max_dist):
        (i, j) = self.discretize_state(agent.state.p_pos)
        mask = np.logical_and(self.apsp[i, j] >= min_dist, self.apsp[i, j] <= max_dist)
        mask = np.logical_and(mask, world.maze_walls == 0)
        candidate_states = np.where(mask)
        num_candidate_states = len(candidate_states[0])
        if num_candidate_states == 0:
            return (agent.state.p_pos, None)
        goal_index = np.random.choice(num_candidate_states)
        goal = np.array(
            [candidate_states[0][goal_index], candidate_states[1][goal_index]],
            dtype=np.float32,
        )
        goal += np.random.uniform(size=2)
        dist_to_goal = self.get_distance(agent.state.p_pos, goal)
        assert min_dist <= dist_to_goal <= max_dist
        assert not self.is_blocked(goal, world)
        return self.normalize_obs(goal)

    def sample_goal(self, agent, world):
        if np.random.random() < self._prob_constraint:
            return self.sample_goal_constrained(
                agent, world, self._min_dist, self._max_dist
            )
        else:
            return self.sample_goal_unconstrained(agent, world)

    @property
    def max_goal_dist(self):
        apsp = self.apsp
        return np.max(apsp[np.isfinite(apsp)])

    def reset_world(self, world):
        # Random properties for agents
        world.assign_agent_colors()
        world.assign_landmark_colors()

        self.set_difficulty(self._difficulty)

        # Set random initial states
        for agent in world.agents:
            agent.state.p_pos = self.sample_empty_state(world)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.goal = self.sample_goal(agent, world)

        for i, landmark in enumerate(world.landmarks):
            if i >= world.num_agents:
                break
            landmark.state.p_pos = world.agents[i].goal
            landmark.state.p_vel = np.zeros(world.dim_p)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = -1.0 * np.linalg.norm(agent.goal - agent.state.p_pos)

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent) and a != agent:
                    # rew -= 10
                    rew = -1

        # # Agents are penalized for exiting the screen, so that they can be caught by the adversaries
        # def bound(x):
        #     if x < 0.9:
        #         return 0
        #     if x < 1.0:
        #         return (x - 0.9) * 10
        #     return min(np.exp(2 * x - 2), 10)

        # for p in range(world.dim_p):
        #     x = abs(agent.state.p_pos[p])
        #     rew -= bound(x)

        return rew

    def observation(self, agent, world):
        # Get positions of all entities in this agent's reference frame
        entity_pos = []
        entity_pos.append(agent.goal - agent.state.p_pos)

        # Communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate(
            [agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm
        )

    def done(self, agent, world, current_step):

        reached_goal = (
            np.linalg.norm(agent.goal - agent.state.p_pos) < self._threshold_distance
        )
        out_of_bounds = (
            agent.state.p_pos[0] < 0
            or agent.state.p_pos[0] >= 1.0
            or agent.state.p_pos[1] < 0
            or agent.state.p_pos[1] >= 1.0
        )
        timeout = current_step >= world.world_length
        return reached_goal or out_of_bounds or timeout
