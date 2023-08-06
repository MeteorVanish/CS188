# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodList = newFood.asList()
        foodDistanceMin = min(map(lambda xy: manhattanDistance(newPos, xy), foodList)) if len(foodList) else 0
        return successorGameState.getScore() * 10 - foodDistanceMin

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        action, value = self.minimax(gameState)
        return action

    def minimax(self, gameState, depth=0, agentIndex=0):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return None, self.evaluationFunction(gameState)
        if agentIndex:  # ghost: min-value
            actionList = gameState.getLegalActions(agentIndex)
            minValue = float('inf')
            if agentIndex == gameState.getNumAgents() - 1:
                nextAgentIndex = 0
                depth += 1
            else:
                nextAgentIndex = agentIndex + 1
            bestAction = actionList[0]
            for action in actionList:
                nextState = gameState.generateSuccessor(agentIndex, action)
                _, value = self.minimax(nextState, depth, nextAgentIndex)
                if value < minValue:
                    minValue = value
                    bestAction = action
            return bestAction, minValue
        else:  # pacman: max-value
            actionList = gameState.getLegalActions(0)
            maxValue = float('-inf')
            bestAction = actionList[0]
            for action in actionList:
                nextState = gameState.generateSuccessor(0, action)
                _, value = self.minimax(nextState, depth, 1)
                if value > maxValue:
                    maxValue = value
                    bestAction = action
            return bestAction, maxValue

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        action, value = self.minimaxAlphaBetaPruning(gameState)
        return action

    def minimaxAlphaBetaPruning(self, gameState, depth=0, agentIndex=0, alpha=float('-inf'), beta=float('inf')):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return None, self.evaluationFunction(gameState)
        if agentIndex:  # ghost: min-value
            actionList = gameState.getLegalActions(agentIndex)
            minValue = float('inf')
            if agentIndex == gameState.getNumAgents() - 1:
                nextAgentIndex = 0
                depth += 1
            else:
                nextAgentIndex = agentIndex + 1
            bestAction = actionList[0]
            for action in actionList:
                nextState = gameState.generateSuccessor(agentIndex, action)
                _, value = self.minimaxAlphaBetaPruning(nextState, depth, nextAgentIndex, alpha, beta)
                if value < minValue:
                    minValue = value
                    bestAction = action
                if minValue < alpha:
                    return bestAction, minValue
                beta = min(beta, minValue)
            return bestAction, minValue
        else:  # pacman: max-value
            actionList = gameState.getLegalActions(0)
            maxValue = float('-inf')
            bestAction = actionList[0]
            for action in actionList:
                nextState = gameState.generateSuccessor(0, action)
                _, value = self.minimaxAlphaBetaPruning(nextState, depth, 1, alpha, beta)
                if value > maxValue:
                    maxValue = value
                    bestAction = action
                if maxValue > beta:
                    return bestAction, maxValue
                alpha = max(alpha, maxValue)
            return bestAction, maxValue

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        action, value = self.expectimax(gameState)
        return action

    def expectimax(self, gameState, depth=0, agentIndex=0):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return None, self.evaluationFunction(gameState)
        if agentIndex:  # ghost
            actionList = gameState.getLegalActions(agentIndex)
            minValue = float('inf')
            if agentIndex == gameState.getNumAgents() - 1:
                nextAgentIndex = 0
                depth += 1
            else:
                nextAgentIndex = agentIndex + 1
            valueSum = 0
            for action in actionList:
                nextState = gameState.generateSuccessor(agentIndex, action)
                _, value = self.expectimax(nextState, depth, nextAgentIndex)
                valueSum += value
            return None, valueSum / len(actionList)
        else:  # pacman
            actionList = gameState.getLegalActions(0)
            maxValue = float('-inf')
            bestAction = actionList[0]
            for action in actionList:
                nextState = gameState.generateSuccessor(0, action)
                _, value = self.expectimax(nextState, depth, 1)
                if value > maxValue:
                    maxValue = value
                    bestAction = action
            return bestAction, maxValue

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()

    foodList = currentGameState.getFood().asList()
    foodDistanceMin = min(map(lambda xy: manhattanDistance(pos, xy), foodList)) if len(foodList) else 0

    ghostEval = 0
    ghostDistanceMin = float('inf')
    for ghost in currentGameState.getGhostStates():
        ghostDistance = manhattanDistance(pos, ghost.getPosition())
        if ghost.scaredTimer == 0:
            ghostDistanceMin = min(ghostDistanceMin, ghostDistance)
        elif ghost.scaredTimer > ghostDistance:
            ghostEval += 100
    if ghostDistanceMin != float('inf'):
        ghostEval += ghostDistanceMin

    return currentGameState.getScore() * 10 - 3 * foodDistanceMin + ghostEval

# Abbreviation
better = betterEvaluationFunction
