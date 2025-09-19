"""
Implement your AI here
Do not change the API signatures for __init__ or __call__
__call__ must return a valid action
"""
import numpy as np
import gomoku as gm

class Submission:
    def __init__(self, board_size, win_size,
                 max_depth=2, neighbor_radius=2, top_k=6):
        self.N = board_size                  # Board size (N x N)
        self.win_size = win_size            # Number of consecutive stones needed to win
        self.max_depth = max_depth          # Maximum depth for minimax full search
        self.radius = neighbor_radius       # Radius around stones to consider for move generation
        self.top_k = top_k                  # Number of top moves for deep search
        self.top_k2 = top_k * 2             # Extended set of moves for shallow search

    def __call__(self, state):
        valid = state.valid_actions()       # All currently valid moves
        # Step 1: Check for immediate winning move or block opponent's win
        for mv in valid:
            nxt = state.perform(mv)
            if nxt.is_game_over() or nxt.current_score() > 0:
                return mv                   # Return immediately winning/blocking move
        # Step 2: Look ahead for tactical win/block in one move
        score, act = self.look_ahead(state)
        if score != 0:
            return act
        # Step 3: Generate candidate moves based on neighbors
        moves = self.candidate_moves(state)
        max_turn = state.is_max_turn()
        # Order candidate moves using heuristic score to help alpha-beta pruning
        moves.sort(key=lambda mv: self.heuristic(state.perform(mv)), reverse=max_turn)
        best_mv = moves[0]
        best_val = float('-inf') if max_turn else float('inf')
        alpha, beta = float('-inf'), float('inf')
        # Step 4: Run minimax with tiered search depths
        for i, mv in enumerate(moves):
            nxt = state.perform(mv)
            if i < self.top_k:
                depth = self.max_depth      # Full depth search for top moves
            elif i < self.top_k2:
                depth = 1                   # Shallow search for next-tier moves
            else:
                depth = 0                   # Heuristic only for others
            v = self.minimax(nxt, depth, max_turn, alpha, beta)
            if max_turn:
                if v > best_val:
                    best_val, best_mv = v, mv
                alpha = max(alpha, best_val)
            else:
                if v < best_val:
                    best_val, best_mv = v, mv
                beta = min(beta, best_val)
            if beta <= alpha:               # Prune the search
                break
        return best_mv

    def candidate_moves(self, state):
        b = state.board
        occ = b[gm.MAX] + b[gm.MIN]         # Board positions that are occupied
        empt = b[gm.EMPTY]                  # Board positions that are empty
        coords = sorted([tuple(xy) for xy in np.argwhere(occ == 1)])  # All occupied positions
        if not coords:
            c = self.N // 2
            return [(c, c)]                 # Play center on empty board
        cand = []
        r = self.radius
        # Look around each occupied position within radius
        for x, y in coords:
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    xx, yy = x+dx, y+dy
                    if 0 <= xx < self.N and 0 <= yy < self.N and empt[xx, yy] == 1:
                        mv = (xx, yy)
                        if mv not in cand:
                            cand.append(mv)
        return cand or sorted(state.valid_actions())  # Fallback if no neighbors found

    def minimax(self, state, depth, maximizing, alpha, beta):
        # Terminal condition for recursion
        if depth <= 0 or state.is_game_over():
            return self.heuristic(state)
        # Skip deep search if not enough turns remain
        if self.turn_bound(state) > depth:
            return self.heuristic(state)
        moves = self.candidate_moves(state)
        if maximizing:
            val = float('-inf')
            for mv in moves:
                v = self.minimax(state.perform(mv), depth-1, False, alpha, beta)
                val = max(val, v)
                alpha = max(alpha, val)
                if alpha >= beta:
                    break                   # Alpha-beta pruning
            return val
        else:
            val = float('inf')
            for mv in moves:
                v = self.minimax(state.perform(mv), depth-1, True, alpha, beta)
                val = min(val, v)
                beta = min(beta, val)
                if beta <= alpha:
                    break                   # Alpha-beta pruning
            return val

    def heuristic(self, state):
        b = state.board
        # Difference in line potential between MAX and MIN
        diff = self.line_score(b, gm.MAX) - self.line_score(b, gm.MIN)
        empt = b[gm.EMPTY].sum()
        weight = 1 if empt > (self.N*self.N)/2 else 2  # Weight increases as board fills
        return weight * diff

    def line_score(self, b, player):
        score = 0
        for dx, dy in [(0,1),(1,0),(1,1),(1,-1)]:  # All 4 directions
            for x in range(self.N):
                for y in range(self.N):
                    if b[player, x, y] == 1:
                        cnt, blk = 1, 2
                        # Check before and after the line
                        bx, by = x-dx, y-dy
                        if 0<=bx<self.N and 0<=by<self.N and b[player,bx,by]==0:
                            blk -= 1
                        fx, fy = x+dx, y+dy
                        if 0<=fx<self.N and 0<=fy<self.N and b[player,fx,fy]==0:
                            blk -= 1
                        # Count consecutive stones
                        nx, ny = x+dx, y+dy
                        while 0<=nx<self.N and 0<=ny<self.N and b[player,nx,ny]==1:
                            cnt += 1; nx += dx; ny += dy
                        # Score based on number of stones and openness
                        if cnt >= self.win_size:
                            score += 100000
                        elif cnt == self.win_size - 1:
                            score += 1000 if blk < 2 else 100
                        elif cnt == self.win_size - 2:
                            score += 100 if blk < 2 else 10
                        else:
                            score += cnt
        return score

    def turn_bound(self, state):
        corr = state.corr                   # Correlation matrix from the state
        empt = state.board[gm.EMPTY].sum()
        is_max = state.is_max_turn()
        bound = empt
        # Find threats with one empty cell for each player
        min_rt = (corr[:,gm.EMPTY] + corr[:,gm.MIN] == self.win_size)
        max_rt = (corr[:,gm.EMPTY] + corr[:,gm.MAX] == self.win_size)
        # Estimate number of turns to win
        min_t = 2*corr[:,gm.EMPTY] - (0 if is_max else 1)
        max_t = 2*corr[:,gm.EMPTY] - (1 if is_max else 0)
        if min_rt.any(): bound = min(bound, int(min_t[min_rt].min()))
        if max_rt.any(): bound = min(bound, int(max_t[max_rt].min()))
        return bound

    def look_ahead(self, state):
        p = state.current_player()
        sign = 1 if p == gm.MAX else -1
        empt = state.board[gm.EMPTY].sum()
        corr = state.corr
        # Check for winning move for self
        idx = np.argwhere((corr[:,gm.EMPTY] == 1) & (corr[:,p] == self.win_size-1))
        if idx.size: return sign * empt, self.find_empty(state, *idx[0])
        # Check for winning move for opponent
        opp = gm.MIN if p == gm.MAX else gm.MAX
        idx2 = np.argwhere((corr[:,gm.EMPTY] == 1) & (corr[:,opp] == self.win_size-1))
        if idx2.size: return -sign * (empt - 1), self.find_empty(state, *idx2[0])
        return 0, None

    def find_empty(self, state, p, r, c):
        b = state.board; n = self.win_size
        # Return empty cell based on direction p
        if p == 0: return r, c + b[gm.EMPTY, r, c:c+n].argmax()
        if p == 1: return r + b[gm.EMPTY, r:r+n, c].argmax(), c
        rng = np.arange(n)
        if p == 2: off = b[gm.EMPTY, r+rng, c+rng].argmax(); return r+off, c+off
        if p == 3: off = b[gm.EMPTY, r-rng, c+rng].argmax(); return r-off, c+off
        return None

if __name__ == "__main__":
    pass
