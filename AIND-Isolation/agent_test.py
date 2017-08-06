"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest

import isolation
import game_agent
from sample_players import HumanPlayer, GreedyPlayer

from importlib import reload


class IsolationTest(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(game_agent)
        self.player1 = game_agent.MinimaxPlayer()
        self.player2 = game_agent.AlphaBetaPlayer()
        self.game = isolation.Board(self.player1, self.player2)

    def testNodeExpansion(self):
        move = self.player1.get_move(self.game, lambda: 2000)
        self.assertEqual(self.player1.expanded, 13681)

    def testAlphaBetaPrune(self):
        move = self.player2.get_move(self.game, lambda: 2000)

if __name__ == '__main__':
    unittest.main()
