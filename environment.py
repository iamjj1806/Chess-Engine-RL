import numpy as np
import chess

class ChessEnvironment:
    """Chess environment that handles board state and legal moves with enhanced rewards."""
    
    # Piece values (standard chess piece values)
    PIECE_VALUES = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.0,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
        chess.KING: 0.0  # King has no material value as it can't be captured
    }
    
    # Position evaluation tables for piece-square values
    # These tables encourage pieces to move to strategically advantageous squares
    PAWN_TABLE = np.array([
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
        [ 5.0,  5.0,  5.0,  5.0,  5.0,  5.0,  5.0,  5.0],
        [ 1.0,  1.0,  2.0,  3.0,  3.0,  2.0,  1.0,  1.0],
        [ 0.5,  0.5,  1.0,  2.5,  2.5,  1.0,  0.5,  0.5],
        [ 0.0,  0.0,  0.0,  2.0,  2.0,  0.0,  0.0,  0.0],
        [ 0.5, -0.5, -1.0,  0.0,  0.0, -1.0, -0.5,  0.5],
        [ 0.5,  1.0,  1.0, -2.0, -2.0,  1.0,  1.0,  0.5],
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]
    ]) * 0.1
    
    KNIGHT_TABLE = np.array([
        [-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0],
        [-4.0, -2.0,  0.0,  0.0,  0.0,  0.0, -2.0, -4.0],
        [-3.0,  0.0,  1.0,  1.5,  1.5,  1.0,  0.0, -3.0],
        [-3.0,  0.5,  1.5,  2.0,  2.0,  1.5,  0.5, -3.0],
        [-3.0,  0.0,  1.5,  2.0,  2.0,  1.5,  0.0, -3.0],
        [-3.0,  0.5,  1.0,  1.5,  1.5,  1.0,  0.5, -3.0],
        [-4.0, -2.0,  0.0,  0.5,  0.5,  0.0, -2.0, -4.0],
        [-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0]
    ]) * 0.1
    
    BISHOP_TABLE = np.array([
        [-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0],
        [-1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0],
        [-1.0,  0.0,  0.5,  1.0,  1.0,  0.5,  0.0, -1.0],
        [-1.0,  0.5,  0.5,  1.0,  1.0,  0.5,  0.5, -1.0],
        [-1.0,  0.0,  1.0,  1.0,  1.0,  1.0,  0.0, -1.0],
        [-1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, -1.0],
        [-1.0,  0.5,  0.0,  0.0,  0.0,  0.0,  0.5, -1.0],
        [-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0]
    ]) * 0.1
    
    ROOK_TABLE = np.array([
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
        [ 0.5,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  0.5],
        [-0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
        [-0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
        [-0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
        [-0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
        [-0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5],
        [ 0.0,  0.0,  0.5,  1.0,  1.0,  0.5,  0.0,  0.0]
    ]) * 0.1
    
    QUEEN_TABLE = np.array([
        [-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0],
        [-1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0],
        [-1.0,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -1.0],
        [-0.5,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -0.5],
        [ 0.0,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -0.5],
        [-1.0,  0.5,  0.5,  0.5,  0.5,  0.5,  0.0, -1.0],
        [-1.0,  0.0,  0.5,  0.0,  0.0,  0.0,  0.0, -1.0],
        [-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0]
    ]) * 0.1
    
    KING_TABLE_MIDDLEGAME = np.array([
        [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
        [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
        [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
        [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
        [-2.0, -3.0, -3.0, -4.0, -4.0, -3.0, -3.0, -2.0],
        [-1.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.0],
        [ 2.0,  2.0,  0.0,  0.0,  0.0,  0.0,  2.0,  2.0],
        [ 2.0,  3.0,  1.0,  0.0,  0.0,  1.0,  3.0,  2.0]
    ]) * 0.1
    
    KING_TABLE_ENDGAME = np.array([
        [-5.0, -4.0, -3.0, -2.0, -2.0, -3.0, -4.0, -5.0],
        [-3.0, -2.0, -1.0,  0.0,  0.0, -1.0, -2.0, -3.0],
        [-3.0, -1.0,  2.0,  3.0,  3.0,  2.0, -1.0, -3.0],
        [-3.0, -1.0,  3.0,  4.0,  4.0,  3.0, -1.0, -3.0],
        [-3.0, -1.0,  3.0,  4.0,  4.0,  3.0, -1.0, -3.0],
        [-3.0, -1.0,  2.0,  3.0,  3.0,  2.0, -1.0, -3.0],
        [-3.0, -3.0,  0.0,  0.0,  0.0,  0.0, -3.0, -3.0],
        [-5.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -5.0]
    ]) * 0.1
    
    def __init__(self, fen=None):
        self.board = chess.Board(fen) if fen else chess.Board()
        self.last_move = None  # Track the last move
        self.prev_score = self.evaluate_position()  # Track previous position score for incremental rewards
    
    def reset(self):
        """Reset the board to the starting position."""
        self.board = chess.Board()
        self.last_move = None  # Reset last move
        self.prev_score = self.evaluate_position()  # Reset previous score
        return self.get_observation()
    
    def get_observation(self):
        """Convert the board to a 8x8x14 observation."""
        observation = np.zeros((8, 8, 14), dtype=np.float32)
        
        # Piece planes (6 piece types x 2 colors)
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece:
                color = 0 if piece.color == chess.WHITE else 1
                piece_type = piece.piece_type - 1  # 0-5 for pawn, knight, bishop, rook, queen, king
                observation[i // 8, i % 8, piece_type + color * 6] = 1.0
        
        # Auxiliary planes
        # Player to move
        if self.board.turn == chess.WHITE:
            observation[:, :, 12] = 1.0
        else:
            observation[:, :, 13] = 1.0
        
        return observation
    
    def is_game_over(self):
        """Check if the game is over."""
        return self.board.is_game_over()
    
    def step(self, move):
        """Make a move on the board and return the new state, reward, and done flag."""
        if isinstance(move, str):
            move = chess.Move.from_uci(move)
        
        if move in self.board.legal_moves:
            # Store previous position score
            prev_score = self.prev_score
            
            # Make the move
            self.board.push(move)
            self.last_move = move  # Store the last move
            
            # Calculate new position score
            new_score = self.evaluate_position()
            self.prev_score = new_score
            
            # Calculate reward based on position improvement and game state
            reward = self.get_reward(prev_score, new_score)
            
            return self.get_observation(), reward, self.is_game_over()
        else:
            print(f"Illegal move: {move}")
            return self.get_observation(), -1, True
    
    def get_reward(self, prev_score=None, new_score=None):
        """Get the reward for the current state."""
        # Terminal rewards (these take precedence)
        if self.board.is_checkmate():
            return 10.0 if self.board.turn == chess.BLACK else -10.0
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_fifty_moves():
            # Slight punishment for draws to encourage decisive play
            return -0.1
        
        # If we have previous and new scores, calculate incremental reward
        if prev_score is not None and new_score is not None:
            # Positive reward for improving position, negative for worsening
            # We negate for black's turn because the evaluation is always from white's perspective
            perspective = 1.0 if self.board.turn == chess.BLACK else -1.0
            incremental_reward = perspective * (new_score - prev_score)
            
            # Add small rewards for specific actions
            additional_reward = 0.0
            
            # Reward for checking the opponent
            if self.board.is_check():
                additional_reward += 0.1
                
            # Small reward for castling (strategic advantage)
            if self.last_move and self.last_move.from_square == chess.E1 and self.last_move.to_square in [chess.C1, chess.G1]:
                additional_reward += 0.3  # White castling
            elif self.last_move and self.last_move.from_square == chess.E8 and self.last_move.to_square in [chess.C8, chess.G8]:
                additional_reward += 0.3  # Black castling
            
            # Return the combined reward
            return incremental_reward + additional_reward
        
        # Default reward if no previous score is available
        return 0.0
    
    def get_result(self):
        """Get the result of the game (1 for white win, -1 for black win, 0 for draw)."""
        if self.board.is_checkmate():
            return -1 if self.board.turn == chess.WHITE else 1
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_fifty_moves():
            return 0
        else:
            return 0
    
    def evaluate_position(self):
        """Evaluate the current position from white's perspective."""
        if self.board.is_checkmate():
            # Checkmate has the highest/lowest possible score
            return -100.0 if self.board.turn == chess.WHITE else 100.0
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_fifty_moves():
            return 0.0  # Draw has a neutral score
        
        # Initialize score
        score = 0.0
        
        # Count material for both sides
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is None:
                continue
                
            value = self.PIECE_VALUES[piece.piece_type]
            
            # Add piece-square value based on position
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            
            # Flip the tables for black pieces
            position_rank = rank if piece.color == chess.WHITE else 7 - rank
            position_file = file
            
            # Add position value based on piece type
            if piece.piece_type == chess.PAWN:
                value += self.PAWN_TABLE[position_rank][position_file]
            elif piece.piece_type == chess.KNIGHT:
                value += self.KNIGHT_TABLE[position_rank][position_file]
            elif piece.piece_type == chess.BISHOP:
                value += self.BISHOP_TABLE[position_rank][position_file]
            elif piece.piece_type == chess.ROOK:
                value += self.ROOK_TABLE[position_rank][position_file]
            elif piece.piece_type == chess.QUEEN:
                value += self.QUEEN_TABLE[position_rank][position_file]
            elif piece.piece_type == chess.KING:
                # Use different tables for middlegame and endgame
                is_endgame = self.is_endgame()
                if is_endgame:
                    value += self.KING_TABLE_ENDGAME[position_rank][position_file]
                else:
                    value += self.KING_TABLE_MIDDLEGAME[position_rank][position_file]
            
            # Adjust score based on piece color
            score += value if piece.color == chess.WHITE else -value
        
        # Mobility: count legal moves (encourages piece development and board control)
        current_turn = self.board.turn
        num_legal_moves = len(list(self.board.legal_moves))
        
        # Switch turn to count opponent's moves
        self.board.turn = not self.board.turn
        num_opponent_moves = len(list(self.board.legal_moves))
        self.board.turn = current_turn  # Switch back
        
        # Add mobility component to score (0.01 per move advantage)
        mobility_score = 0.01 * (num_legal_moves - num_opponent_moves)
        score += mobility_score if current_turn == chess.WHITE else -mobility_score
        
        # Pawn structure evaluation
        score += self.evaluate_pawn_structure()
        
        # King safety
        score += self.evaluate_king_safety()
        
        return score
    
    def is_endgame(self):
        """Determine if the position is in the endgame phase."""
        # Count material on the board
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is None:
                continue
                
            value = self.PIECE_VALUES[piece.piece_type]
            if piece.color == chess.WHITE:
                white_material += value
            else:
                black_material += value
        
        # An endgame is typically when both sides have <= 13 points of material (excluding kings)
        # or when one side has only a queen or less
        return (white_material <= 13 and black_material <= 13) or white_material <= 9 or black_material <= 9
    
    def evaluate_pawn_structure(self):
        """Evaluate pawn structure strengths and weaknesses."""
        score = 0.0
        
        # Detect doubled pawns (pawns on the same file)
        for file in range(8):
            white_pawns = 0
            black_pawns = 0
            for rank in range(8):
                square = chess.square(file, rank)
                piece = self.board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN:
                    if piece.color == chess.WHITE:
                        white_pawns += 1
                    else:
                        black_pawns += 1
            
            # Penalize doubled pawns
            if white_pawns > 1:
                score -= 0.2 * (white_pawns - 1)
            if black_pawns > 1:
                score += 0.2 * (black_pawns - 1)
        
        # Detect isolated pawns (no friendly pawns on adjacent files)
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                file = chess.square_file(square)
                has_neighbor = False
                
                # Check adjacent files
                for adj_file in [file - 1, file + 1]:
                    if adj_file < 0 or adj_file > 7:
                        continue
                    
                    for rank in range(8):
                        adj_square = chess.square(adj_file, rank)
                        adj_piece = self.board.piece_at(adj_square)
                        if adj_piece and adj_piece.piece_type == chess.PAWN and adj_piece.color == piece.color:
                            has_neighbor = True
                            break
                
                # Penalize isolated pawns
                if not has_neighbor:
                    score -= 0.3 if piece.color == chess.WHITE else -0.3
        
        # Reward passed pawns (no enemy pawns ahead on the same file or adjacent files)
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                is_passed = True
                
                direction = 1 if piece.color == chess.WHITE else -1
                enemy_color = not piece.color
                
                # Check for enemy pawns that could block or capture
                for check_file in [file - 1, file, file + 1]:
                    if check_file < 0 or check_file > 7:
                        continue
                    
                    for check_rank in range(rank + direction, 8 if direction > 0 else -1, direction):
                        if check_rank < 0 or check_rank > 7:
                            continue
                            
                        check_square = chess.square(check_file, check_rank)
                        check_piece = self.board.piece_at(check_square)
                        if check_piece and check_piece.piece_type == chess.PAWN and check_piece.color == enemy_color:
                            is_passed = False
                            break
                
                # Reward passed pawns (more for advanced pawns)
                if is_passed:
                    advanced_rank = rank if piece.color == chess.WHITE else 7 - rank
                    bonus = 0.2 + 0.1 * advanced_rank  # Higher bonus for more advanced pawns
                    score += bonus if piece.color == chess.WHITE else -bonus
        
        return score
    
    def evaluate_king_safety(self):
        """Evaluate king safety based on surrounding squares and pawn shield."""
        score = 0.0
        
        # Evaluate king safety for both colors
        for color in [chess.WHITE, chess.BLACK]:
            king_square = self.board.king(color)
            if king_square is None:  # Shouldn't happen in a valid position
                continue
                
            king_file = chess.square_file(king_square)
            king_rank = chess.square_rank(king_square)
            
            # Count enemy attacks near the king
            attack_count = 0
            defenders_count = 0
            
            # Check squares around the king
            for file_offset in [-1, 0, 1]:
                for rank_offset in [-1, 0, 1]:
                    if file_offset == 0 and rank_offset == 0:
                        continue  # Skip the king's square itself
                        
                    check_file = king_file + file_offset
                    check_rank = king_rank + rank_offset
                    
                    if check_file < 0 or check_file > 7 or check_rank < 0 or check_rank > 7:
                        continue
                        
                    check_square = chess.square(check_file, check_rank)
                    
                    # Check if square is attacked by enemy
                    is_attacked = self.board.is_attacked_by(not color, check_square)
                    if is_attacked:
                        attack_count += 1
                    
                    # Check if square is defended by friendly piece
                    defending_piece = self.board.piece_at(check_square)
                    if defending_piece and defending_piece.color == color:
                        defenders_count += 1
            
            # Penalize exposed king (penalty depends on phase of the game)
            is_endgame = self.is_endgame()
            if not is_endgame:  # In middlegame, king safety is more important
                safety_penalty = 0.1 * attack_count - 0.05 * defenders_count
                score -= safety_penalty if color == chess.WHITE else -safety_penalty
            
            # Check for pawn shield in front of castled king
            if not is_endgame and ((color == chess.WHITE and king_rank == 0) or 
                                   (color == chess.BLACK and king_rank == 7)):
                pawn_shield_count = 0
                pawn_rank = 1 if color == chess.WHITE else 6
                
                # Check for pawns in front of the king
                for file in range(max(0, king_file - 1), min(8, king_file + 2)):
                    shield_square = chess.square(file, pawn_rank)
                    shield_piece = self.board.piece_at(shield_square)
                    if shield_piece and shield_piece.piece_type == chess.PAWN and shield_piece.color == color:
                        pawn_shield_count += 1
                
                # Reward for having pawns as a shield
                shield_bonus = 0.1 * pawn_shield_count
                score += shield_bonus if color == chess.WHITE else -shield_bonus
        
        return score