from copy import deepcopy
import time
import math
import random


class GoBangState:
    def __init__(self, board, currentPlayer=1, last_move=None):
        if last_move is None:
            last_move = [0, 0]
        self.board = board  # 五子棋棋盘
        self.currentPlayer = currentPlayer  # 执黑还是执白，1是黑，-1是白
        self.last_move = last_move  # 上一手棋的位置

    """
     搜索空间太大
     possibleActions = []
     for i in range(len(self.board)):
          for j in range(len(self.board[i])):
                 if self.board[i][j] == 0:
                      possibleActions.append(Action(player=self.currentPlayer, x=i, y=j))
     return possibleActions
    """

    # 改成在上一手棋周围进行搜索
    def getPossibleActions(self):
        possibleActions = []
        search_size = 1
        found = False
        while not found and len(possibleActions) == 0:
            for i in range(max(0, self.last_move[0] - search_size), min(len(self.board), self.last_move[0] + search_size + 1)):
                for j in range(max(0, self.last_move[1] - search_size), min(len(self.board[i]), self.last_move[1] + search_size + 1)):
                    if self.board[i][j] == 0:
                        possibleActions.append(Action(player=self.currentPlayer, x=i, y=j))
                        found = True
            search_size += 1
        return possibleActions

    def takeAction(self, action):
        newState = deepcopy(self)
        newState.board[action.x][action.y] = action.player
        newState.currentPlayer = self.currentPlayer * -1
        newState.last_move = [action.x, action.y]
        return newState

    def isTerminal(self):
        # 检查是否有一方胜利
        if judge(self.board) == 1:
            return True
        if judge(self.board) == -1:
            return True
        # 检查是否平局（棋盘已满且无胜利者）
        if all(self.board[i][j] != 0 for i in range(len(self.board)) for j in range(len(self.board[i]))):
            return True
        return False

    def getReward(self):
        # 返回胜利的玩家编号（1或-1），或者0表示平局
        result = judge(self.board)
        if result == 1:
            return 1
        elif result == -1:
            return -1
        else:
            return 0

    def getCurrentPlayer(self):
        return self.currentPlayer


class Action:
    def __init__(self, player, x, y):
        self.player = player
        self.x = x
        self.y = y

    def __str__(self):
        return str((self.x, self.y))

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.x == other.x and self.y == other.y and self.player == other.player

    def __hash__(self):
        return hash((self.x, self.y, self.player))


# 判断五子相连
def judge(board):
    board_size = len(board)
    # Check horizontal
    for i in range(board_size):
        for j in range(board_size - 4):
            if all(board[i][j+k] == 1 for k in range(5)):
                return 1
            if all(board[i][j+k] == -1 for k in range(5)):
                return -1
    # Check vertical
    for i in range(board_size - 4):
        for j in range(board_size):
            if all(board[i+k][j] == 1 for k in range(5)):
                return 1
            if all(board[i+k][j] == -1 for k in range(5)):
                return -1
    # Check diagonal (top-left to bottom-right)
    for i in range(board_size - 4):
        for j in range(board_size - 4):
            if all(board[i+k][j+k] == 1 for k in range(5)):
                return 1
            if all(board[i+k][j+k] == -1 for k in range(5)):
                return -1
    # Check anti-diagonal (top-right to bottom-left)
    for i in range(4, board_size):
        for j in range(board_size - 4):
            if all(board[i-k][j+k] == 1 for k in range(5)):
                return 1
            if all(board[i-k][j+k] == -1 for k in range(5)):
                return -1
    return 0


# MCTS
def randomPolicy(state):
    while not state.isTerminal():
        try:
            action = random.choice(state.getPossibleActions())
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.takeAction(action)
    return state.getReward()


class treeNode:
    def __init__(self, state, parent):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}

    def __str__(self):
        s = []
        s.append("totalReward: %s" % self.totalReward)
        s.append("numVisits: %d" % self.numVisits)
        s.append("isTerminal: %s" % self.isTerminal)
        s.append("possibleActions: %s" % (self.children.keys()))
        return "%s: {%s}" % (self.__class__.__name__, ', '.join(s))


class mcts:
    def __init__(self, timeLimit=None, iterationLimit=None, explorationConstant=1 / math.sqrt(2), rolloutPolicy=randomPolicy):
        if timeLimit != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy

    def search(self, initialState, needDetails=False):
        self.root = treeNode(initialState, None)
        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:
            for i in range(self.searchLimit):
                self.executeRound()

        bestChild = self.getBestChild(self.root, 0)
        action = (action for action, node in self.root.children.items() if node is bestChild).__next__()
        if needDetails:
            return {"action": action, "expectedReward": bestChild.totalReward / bestChild.numVisits}
        else:
            return action

    def executeRound(self):
        node = self.selectNode(self.root)
        reward = self.rollout(node.state)
        self.backpropogate(node, reward)

    def selectNode(self, node):
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        actions = node.state.getPossibleActions()
        for action in actions:
            if action not in node.children:
                newNode = treeNode(node.state.takeAction(action), node)
                node.children[action] = newNode
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                return newNode

        raise Exception("Should never reach here")

    def backpropogate(self, node, reward):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def getBestChild(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            nodeValue = node.state.getCurrentPlayer() * child.totalReward / child.numVisits + explorationValue * math.sqrt(
                2 * math.log(node.numVisits) / child.numVisits)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)


def main():
    board_size = 15
    state = GoBangState(board=[[0 for _ in range(board_size)] for _ in range(board_size)])
    searcher = mcts(timeLimit=1000)
    move_count = 0
    while not state.isTerminal():
        bestAction = searcher.search(initialState=state, needDetails=False)
        state = state.takeAction(bestAction)
        move_count += 1
        print(f"Move {move_count}: Player {state.getCurrentPlayer()} plays at ({bestAction.x}, {bestAction.y})")
        print(f"Current board:\n{state.board}")
        if state.isTerminal():
            result = state.getReward()
            if result == 1:
                print(f"Game over. Winner is Player 1")
            elif result == -1:
                print(f"Game over. Winner is Player 2")
            else:
                print(f"Game over. It's a draw")
            print(state.board)
            break


if __name__ == "__main__":
    main()
