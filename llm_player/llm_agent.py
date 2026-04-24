import chess
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import re

class SimpleLLM: 
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2-large")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def clean_move(self, move_string):
        matches = re.findall(r'\b[a-h][1-8][a-h][1-8][qrbn]?\b', move_string.lower())
        return matches[-1] if matches else ""

    def causes_repetition(self, board, move, seen_positions=None):
        temp_board = board.copy()
        temp_board.push(move)
        if temp_board.can_claim_draw() or temp_board.is_repetition(2):
            return True
        if seen_positions is not None:
            next_key = LLMAgent.position_key(temp_board)
            if seen_positions.get(next_key, 0) >= 1:
                return True
        return False

    def fallback_move(self, board, seen_positions=None):
        legal_moves = list(board.legal_moves)
        non_repeating = [move for move in legal_moves if not self.causes_repetition(board, move, seen_positions)]
        candidate_pool = non_repeating if non_repeating else legal_moves
        captures = [move for move in candidate_pool if board.is_capture(move)]
        checks = [move for move in candidate_pool if board.gives_check(move)]
        if checks:
            return random.choice(checks)
        elif captures:
            return random.choice(captures)
        else:
            return random.choice(candidate_pool)

    def generate_move(self, board, seen_positions=None, max_attempts=5):
        legal_moves = list(board.legal_moves)
        repetitive_candidates = []

        for attempt in range(max_attempts):
            prompt = f"Chess FEN: {board.fen()}\n"
            prompt += f"Legal moves: {' '.join([move.uci() for move in legal_moves])}\n"
            prompt += "Choose the best move from the legal moves. Respond with only the UCI notation of the chosen move (e.g., e2e4):"
            
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id).float()

            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=5,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            generated_tokens = outputs[0][input_ids.shape[-1]:]
            suggested_move = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            suggested_move = self.clean_move(suggested_move)

            if suggested_move:
                try:
                    move = chess.Move.from_uci(suggested_move)
                    if move in legal_moves:
                        if self.causes_repetition(board, move, seen_positions):
                            repetitive_candidates.append(move)
                            continue
                        return move, f"Valid move {suggested_move} generated on attempt {attempt + 1}."
                except ValueError:
                    pass

        if repetitive_candidates:
            return random.choice(repetitive_candidates), "Only repetition-prone moves were generated; selecting one."

        chosen_move = self.fallback_move(board, seen_positions)
        thoughts = f"Failed to generate a valid move after {max_attempts} attempts. Using fallback strategy."
        return chosen_move, thoughts

class LLMAgent:
    def __init__(self):
        self.move_cache = {}
        self.seen_positions = {}
        self.llm = SimpleLLM()

    @staticmethod
    def position_key(board):
        turn = "w" if board.turn == chess.WHITE else "b"
        castling = board.castling_xfen()
        ep = "-" if board.ep_square is None else chess.square_name(board.ep_square)
        return f"{board.board_fen()} {turn} {castling} {ep}"

    def get_action(self, state, timeout=1):
        board = state if isinstance(state, chess.Board) else chess.Board(state)
        current_key = self.position_key(board)
        self.seen_positions[current_key] = self.seen_positions.get(current_key, 0) + 1

        if current_key in self.move_cache:
            cached_uci = self.move_cache[current_key]
            cached_move = chess.Move.from_uci(cached_uci)
            if (
                cached_move in board.legal_moves
                and self.seen_positions[current_key] == 1
                and not self.llm.causes_repetition(board, cached_move, self.seen_positions)
            ):
                return cached_uci

        chosen_move, thoughts = self.llm.generate_move(board, self.seen_positions)

        move_uci = chosen_move.uci()
        self.move_cache[current_key] = move_uci
        
        print(f"LLM thoughts:\n{thoughts}")
        return move_uci
