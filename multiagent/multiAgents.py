# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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

        minDistanceToGhost = 999
        for i in range (0, len(newGhostStates)):
            if newScaredTimes[i] > 1:
                continue
            ghostPos = newGhostStates[i].getPosition()
            distance = abs(ghostPos[0] - newPos[0]) + abs(ghostPos[1] - newPos[1])
            minDistanceToGhost = min(minDistanceToGhost, distance)

        foods = newFood.asList()
        distancesToFood = [abs(newPos[0] - food[0]) + abs(newPos[1] - food[1]) for food in foods]
        minDistanceToFood = 0
        if len(distancesToFood) != 0:
            minDistanceToFood = min(distancesToFood)

        "*** YOUR CODE HERE ***"
        if minDistanceToGhost < 2:
            return -99999
        score = successorGameState.getScore()
        return score * 10 - minDistanceToFood

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        """

##        actions = gameState.getLegalActions(agentIndex)
##        successors = [gameState.generateSuccessor(agentIndex, action) for action in actions]
##        costs = [self.evaluationFunction(successor) for successor in successors]
##        maxCostIndex = costs.index(max(costs))
        "*** YOUR CODE HERE ***"
##        self.counter = 0
        action, cost = self.getMaxActionAndCost(gameState, self.depth)
        return action

    def getMaxActionAndCost(self, gameState, depth):
        actions = gameState.getLegalActions(0)
        successors = [gameState.generateSuccessor(0, action) for action in actions]
        costs = []
        for s in successors:
            if s.isWin() or s.isLose():
                costs.append(self.evaluationFunction(s))
                continue
            costs.append(self.getMinCost(s, depth, 1))

        maxCost = max(costs)
        maxCostIndex = costs.index(maxCost)
        return (actions[maxCostIndex], maxCost)

    def getMinCost(self, gameState, depth, agentIndex):
        actions = gameState.getLegalActions(agentIndex)
        successors = [gameState.generateSuccessor(agentIndex, action) for action in actions]
        costs = []
        for s in successors:
            if s.isLose() or s.isWin():
                costs.append(self.evaluationFunction(s))
                continue
            cost = 0
            if agentIndex + 1 == gameState.getNumAgents():
                if depth == 1:
                    cost = self.evaluationFunction(s)
                else:
                    cost = self.getMaxActionAndCost(s, depth - 1)[1]
            else:
                cost = self.getMinCost(s, depth, agentIndex + 1)
            costs.append(cost)
        return min(costs)



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        action, cost = self.getMaxActionAndCost(gameState, self.depth, -99999, 99999)
        return action

    def getMaxActionAndCost(self, gameState, depth, a, b):
        actions = gameState.getLegalActions(0)
        maxCost = -99999
        maxAction = actions[0]
        value = 0
        for action in actions:
            s = gameState.generateSuccessor(0, action)
            if s.isWin() or s.isLose():
                value = self.evaluationFunction(s)
            else:
                value = self.getMinCost(s, depth, 1, a, b)
            if value > maxCost:
                maxCost = value
                maxAction = action
            if maxCost > b:
                return (maxAction, maxCost)
            a = max(a, maxCost)

        return (maxAction, maxCost)

    def getMinCost(self, gameState, depth, agentIndex, a, b):
        actions = gameState.getLegalActions(agentIndex)
        minCost = 99999
        for action in actions:
            s = gameState.generateSuccessor(agentIndex, action)
            if s.isLose() or s.isWin():
                value = self.evaluationFunction(s)
            elif agentIndex + 1 == gameState.getNumAgents():
                if depth == 1:
                    value = self.evaluationFunction(s)
                else:
                    value = self.getMaxActionAndCost(s, depth - 1, a, b)[1]
            else:
                value = self.getMinCost(s, depth, agentIndex + 1, a, b)
            minCost = min(minCost, value)
            if minCost < a:
                return minCost
            b = min(b, minCost)

        return minCost

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        action, cost = self.getMaxActionAndCost(gameState, self.depth)
        return action

    def getMaxActionAndCost(self, gameState, depth):
        actions = gameState.getLegalActions(0)
        maxCost = -99999
        maxAction = actions[0]
        value = 0
        for action in actions:
            s = gameState.generateSuccessor(0, action)
            if s.isWin() or s.isLose():
                value = self.evaluationFunction(s)
            else:
                value = self.getAverageCost(s, depth, 1)
            if value > maxCost:
                maxCost = value
                maxAction = action

        return (maxAction, maxCost)

    def getAverageCost(self, gameState, depth, agentIndex):
        actions = gameState.getLegalActions(agentIndex)
        sum = 0
        value = 0
        for action in actions:
            s = gameState.generateSuccessor(agentIndex, action)
            if s.isLose() or s.isWin():
                value = self.evaluationFunction(s)
            elif agentIndex + 1 == gameState.getNumAgents():
                if depth == 1:
                    value = self.evaluationFunction(s)
                else:
                    value = self.getMaxActionAndCost(s, depth - 1)[1]
            else:
                value = self.getAverageCost(s, depth, agentIndex + 1)
            sum += value

        return float(sum) / float(len(actions))

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
##        successorGameState = currentGameState.generatePacmanSuccessor(action)
##    return currentGameState.getScore()

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    minDistanceToScaredGhost = 30
    minDistanceToGhost = 999
    for i in range (0, len(newGhostStates)):
        ghostPos = newGhostStates[i].getPosition()
        distance = abs(ghostPos[0] - newPos[0]) + abs(ghostPos[1] - newPos[1])
        if newScaredTimes[i] > distance / 2:
            minDistanceToScaredGhost = min(minDistanceToScaredGhost, distance)
        else:
            minDistanceToGhost = min(minDistanceToGhost, distance)

    foods = newFood.asList()
    distancesToFood = [abs(newPos[0] - food[0]) + abs(newPos[1] - food[1]) for food in foods]
    minDistanceToFood = 0
    if len(distancesToFood) != 0:
        minDistanceToFood = min(distancesToFood)

    if minDistanceToGhost < 2:
        return -99999
    score = currentGameState.getScore()
    return  score - minDistanceToFood - len(foods) * 20 - minDistanceToScaredGhost

def huntingEvaluationFunction(currentGameState):
    capsules = currentGameState.getCapsules()
    pos = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    minDistanceToScaredGhost = 30
    minDistanceToGhost = 999
    for i in range (0, len(ghostStates)):
        ghostPos = ghostStates[i].getPosition()
        distance = abs(ghostPos[0] - pos[0]) + abs(ghostPos[1] - pos[1])
        if scaredTimes[i] > 1:
            minDistanceToScaredGhost = min(minDistanceToScaredGhost, distance)
        else:
            minDistanceToGhost = min(minDistanceToGhost, distance)

    if minDistanceToGhost < 2:
        return -99999

    minDistanceToCapsule = 0
    if capsules != []:
        minDistanceToCapsule = min([(abs(c[0] - pos[0]) + abs(c[1] - pos[1])) for c in capsules])

    scaredBonus = 0
    if max(scaredTimes) != 0:
        scaredBonus = 100
        minDistanceToCapsule = 0

    distancesToFood = [abs(pos[0] - food[0]) + abs(pos[1] - food[1]) for food in foods]
    minDistanceToFood = 0
    if len(distancesToFood) != 0 and len(capsules) != 0:
        minDistanceToFood = min(distancesToFood)

    score = currentGameState.getScore()
##    foodVar = 0
##    if len(capsules) == 0:
    foodVar = (minDistanceToFood + len(foods)*30) * -1
    return score + scaredBonus - minDistanceToCapsule - minDistanceToScaredGhost + foodVar

def expectimaxEvaluationFunction(currentGameState):
    pass


# Abbreviation
better = huntingEvaluationFunction #betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        action, cost = self.getMaxActionAndCost(gameState, self.depth, -99999, 99999)
        return action

    def getMaxActionAndCost(self, gameState, depth, a, b):
        actions = gameState.getLegalActions(0)
        maxCost = -99999
        maxAction = actions[0]
        value = 0
        for action in actions:
            s = gameState.generateSuccessor(0, action)
            if s.isWin() or s.isLose():
                value = better(s)
            else:
                value = self.getMinCost(s, depth, 1, a, b)
            if value > maxCost:
                maxCost = value
                maxAction = action
            if maxCost > b:
                return (maxAction, maxCost)
            a = max(a, maxCost)

        return (maxAction, maxCost)

    def getMinCost(self, gameState, depth, agentIndex, a, b):
        actions = gameState.getLegalActions(agentIndex)
        minCost = 99999
        for action in actions:
            s = gameState.generateSuccessor(agentIndex, action)
            if s.isLose() or s.isWin():
                value = better(s)
            elif agentIndex + 1 == gameState.getNumAgents():
                if depth == 1:
                    value = better(s)
                else:
                    value = self.getMaxActionAndCost(s, depth - 1, a, b)[1]
            else:
                value = self.getMinCost(s, depth, agentIndex + 1, a, b)
            minCost = min(minCost, value)
            if minCost < a:
                return minCost
            b = min(b, minCost)

        return minCost

