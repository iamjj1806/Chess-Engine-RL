import numpy as np
import chess
import math
from config import Config
from environment import ChessEnvironment
import tensorflow as tf

class Node:
    """Node in the MCTS tree."""
    
    def __init__(self, state, prior=0.0):
        self.state = state  # Chess environment state
        self.visit_count = 0
        self.prior = prior  # P(s,a) from the neural network
        self.value_sum = 0  # Sum of values (to compute average)
        self.children = {}  # Dict of child nodes, keys are moves

    def expanded(self):
        """Check if this node has been expanded."""
        return len(self.children) > 0
    
    def value(self):
        """Calculate the value of this node."""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class MCTS:
    """Monte Carlo Tree Search algorithm."""
    
    def __init__(self, model, dirichlet_noise=True):
        self.model = model
        self.dirichlet_noise = dirichlet_noise
    
    def search(self, state, num_simulations, temperature=1.0):
        """Perform MCTS search starting from the given state."""
        root = Node(state)

        # Expand root node
        self.expand(root)

        # Add Dirichlet noise to root node probabilities (for exploration)
        if self.dirichlet_noise:
            self.add_dirichlet_noise(root)

        # Perform simulations
        for _ in range(num_simulations):
            node = root
            scratch_state = ChessEnvironment(root.state.board.fen())
            
            # Selection: Traverse the tree until reaching an unexpanded node
            search_path = [node]
            while node.expanded():
                action, node = self.select_child(node)
                try:
                    move = chess.Move.from_uci(action)
                    if move in scratch_state.board.legal_moves:
                        scratch_state.board.push(move)
                        search_path.append(node)
                    else:
                        # If move is not legal, stop the simulation
                        break
                except ValueError:
                    # If move is invalid, stop the simulation
                    break

            # Check if the game is over
            value = 0
            if scratch_state.is_game_over():
                value = scratch_state.get_result()
            else:
                # Ensure the current node has a state
                if node.state is None:
                    node.state = scratch_state
                
                # Expansion: Expand the leaf node if not already expanded
                if not node.expanded():
                    self.expand(node)

                # Evaluation: Use the model to evaluate the leaf node
                value = self.evaluate(node.state)

            # Backup: Update statistics for all visited nodes
            self.backup(search_path, value, scratch_state.board.turn)

        # Choose the best action based on visit counts
        return self.select_action(root, temperature)

    def select_child(self, node):
        """Select the child with the highest UCB score."""
        best_score = -float('inf')
        best_action = None
        best_child = None

        # Calculate UCB for each child
        for action, child in node.children.items():
            ucb_score = self.ucb_score(node, child)
            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action
                best_child = child

        return best_action, best_child

    def ucb_score(self, parent, child):
        """Calculate the UCB score for a child node."""
        prior_score = Config.C_PUCT * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        value_score = 0 if child.visit_count == 0 else child.value()
        return value_score + prior_score

    def expand(self, node):
        """Expand a node by adding all possible children."""
        if node.state is None:
            print("Warning: Attempted to expand a node with no state.")
            return

        legal_moves = [move.uci() for move in node.state.board.legal_moves]
        if not legal_moves:
            return  # No legal moves, do not expand

        policy, _ = self.predict(node.state)

        policy_sum = 0
        for move in legal_moves:
            # Create a new state for the child by applying the move
            child_state = ChessEnvironment(node.state.board.fen())
            try:
                chess_move = chess.Move.from_uci(move)
                child_state.board.push(chess_move)
                
                from_square = chess.parse_square(move[:2])
                to_square = chess.parse_square(move[2:4])
                move_index = from_square * 64 + to_square

                prior = policy[move_index]
                policy_sum += prior
                node.children[move] = Node(state=child_state, prior=prior)  # Initialize with proper state
            except ValueError:
                print(f"Warning: Invalid move {move}")
                continue

        # Normalize priors if we have any valid moves
        if policy_sum > 0:
            for child in node.children.values():
                child.prior /= policy_sum

    def predict(self, state):
        """Use the model to predict policy and value."""
        input_tensor = np.expand_dims(state.get_observation(), axis=0)
        policy, value = self.model(input_tensor, training=False)
        policy = tf.nn.softmax(policy[0]).numpy()
        return policy, value[0][0]

    def evaluate(self, state):
        """Evaluate a leaf node using the model."""
        _, value = self.predict(state)
        return value

    def backup(self, search_path, value, turn):
        """Update value and visit count for all nodes in the search path."""
        for node in reversed(search_path):
            # Skip nodes with no state (should not happen with corrections)
            if node.state is None or node.state.board is None:
                print("Warning: Skipping backup for node with no state.")
                continue
                
            node.value_sum += value if node.state.board.turn == turn else -value
            node.visit_count += 1

    def add_dirichlet_noise(self, node):
        """Add Dirichlet noise to the prior probabilities in the root node."""
        if not node.children:
            return
        
        moves = list(node.children.keys())
        noise = np.random.dirichlet([Config.DIRICHLET_NOISE_ALPHA] * len(moves))
        
        for i, move in enumerate(moves):
            node.children[move].prior = (1 - Config.DIRICHLET_NOISE_EPSILON) * node.children[move].prior + \
                                        Config.DIRICHLET_NOISE_EPSILON * noise[i]

    def select_action(self, root, temperature=1.0):
        """Select an action based on MCTS search statistics."""
        visits = np.array([child.visit_count for child in root.children.values()])
        actions = list(root.children.keys())
        
        if not actions:
            return None, {}  # No legal moves

        if temperature == 0 or len(actions) == 1:
            best_move = actions[np.argmax(visits)]
            return best_move, {a: (1.0 if a == best_move else 0.0) for a in actions}

        visits = visits ** (1.0 / temperature)
        total_visits = np.sum(visits)
        probs = visits / total_visits if total_visits > 0 else np.ones_like(visits) / len(visits)

        move = np.random.choice(actions, p=probs)
        return move, {a: p for a, p in zip(actions, probs)}