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

        "*** YOUR CODE HERE ***"
        #initialing modifiers
        distanceMod = -50
        foodDistanceMod = -3
        foodCountMod = 1000
        
        #retrieving food data and distace to nearest ghost
        food = newFood.asList()
        foodCount = foodCountMod * len(food)
        Ghost_nearest = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
        
        #food distance is either distance to nearest food or infinite
        if food:
            F_Distance = foodDistanceMod * min([manhattanDistance(newPos, f) for f in food])
        else:
            F_Distance = float("inf")
        
        #ghost distance is either distance to nearest ghost or -infinite
        if Ghost_nearest:
            G_Distance = distanceMod/Ghost_nearest
        else:
            G_Distance = -float("inf")

        return F_Distance + G_Distance - foodCount

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
    def minimax(self, gameState, depth, agentIndex, maximizingPlayer):
        # If the state is a terminal state or depth is far enough
        if (depth > self.depth or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState), Directions.STOP
        moves = gameState.getLegalActions(agentIndex)
        if maximizingPlayer: # Pacman
            bestValue = -float('inf')
            values = [self.minimax(gameState.generateSuccessor(agentIndex, move), depth, 1, False)[0] for move in moves]
            bestValue = max(values)
            bestIndices = [index for index in range(len(values)) if values[index] == bestValue]
            chosenIndex = random.choice(bestIndices) #Choose randomly from the best moves
            bestMove = moves[chosenIndex]
            return bestValue, bestMove

        else: # Minimizing player
            bestValue = float('inf')
            if (agentIndex < (gameState.getNumAgents() - 1)): # If the agent is not the last one (last ghost)
                values = [self.minimax(gameState.generateSuccessor(agentIndex, move), depth, agentIndex + 1, False)[0] for move in moves]
            else: # If the agent is the last (the last ghost)
                values = [self.minimax(gameState.generateSuccessor(agentIndex, move), depth + 1, 0, True)[0] for move in moves]
            bestValue = min(values)
            bestIndices = [index for index in range(len(values)) if values[index] == bestValue]
            chosenIndex = random.choice(bestIndices) #Choose randomly from the best moves
            bestMove = moves[chosenIndex]
            return bestValue, bestMove
        
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
        "*** YOUR CODE HERE ***"
        chosenAction = self.minimax(gameState, 1, 0, True)[1]
        return chosenAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def alphabeta(self, gameState, depth, agentIndex):
    
        # Alpha and beta are initialized to -inf/inf
        a = float('-inf')
        b = float('inf')
        # Best value is initialized to -inf
        bestValue = float('-inf')
        # If no better action is calculated, stop is a preferred action
        bestMove = Directions.STOP
        # If the state is a terminal state or the depth is reached
        if (depth <= 0 or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState), bestMove
        # Get all legal moves of the agent
        moves = gameState.getLegalActions(agentIndex)
        if len(moves) == 0: #If there are no legal moves
            return self.evaluationFunction(gameState), bestMove
        # Loop though all actions
        for move in moves:
            bestValue = self.alphabeta_min(gameState.generateSuccessor(agentIndex, move), a, b, depth, 1)[0]
            # Update alpha if it was smaller than the best value and get the best move
            if a < bestValue:
                a = bestValue
                bestMove = move
        return bestValue, bestMove

    def alphabeta_max(self, gameState, a, b, depth, agentIndex):

        # Best value is initialized to -inf first
        bestValue = float('-inf')
        # If no better action is calculated, stop is a preferred action
        bestMove = Directions.STOP
        # If the state is a terminal state or the depth is reached
        if (depth <= 0 or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState), bestMove
        # Get all legal moves of the agent
        moves = gameState.getLegalActions(agentIndex)
        if len(moves) == 0: #If there are no legal moves
            return self.evaluationFunction(gameState), bestMove
        else:
            # Loop through all legal moves
            for move in moves:
                value = self.alphabeta_min(gameState.generateSuccessor(agentIndex, move), a, b, depth, 1)[0]
                bestValue = max(bestValue, value)
                if bestValue > b:
                    bestMove = move
                    return bestValue, bestMove
                # Update alpha
                a = max(a, bestValue)
            return bestValue, bestMove

    def alphabeta_min(self, gameState, a, b, depth, agentIndex):

        # Best value is initialized to inf first
        bestValue = float('inf')
        # If no better action is calculated, stop is a preferred action
        bestMove = Directions.STOP
        # If the state is a terminal state or the depth is reached
        if (depth <= 0 or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState), Directions.STOP
        # Get all legal moves of the agent
        moves = gameState.getLegalActions(agentIndex)
        if len(moves) == 0: # If there are no legal moves
            return self.evaluationFunction(gameState), bestMove
        else:
            # Loop through all legal moves
            for move in moves:
                if (agentIndex < (gameState.getNumAgents() - 1)): # If the agent is not the last one (last ghost)
                    value = self.alphabeta_min(gameState.generateSuccessor(agentIndex, move), a, b, depth, agentIndex + 1)[0]
                else: # If the agent is the last agent (last ghost)
                    value = self.alphabeta_max(gameState.generateSuccessor(agentIndex, move), a, b, depth - 1, 0)[0]
                bestValue = min(bestValue, value)
                if (bestValue < a):
                    bestMove = move
                    return bestValue, bestMove
                # Update beta
                b = min(b, bestValue)
        return bestValue, bestMove

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Initial call for alphabeta function
        chosenAction = self.alphabeta(gameState, self.depth, 0)[1]
        return chosenAction

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
        def expectimax(depth, state, agentIndex):
        
            maxLayer = 0
            
            #in min layer
            if agentIndex == state.getNumAgents():
                if depth == self.depth:
                    #In max depth the state is re-evaluated
                    evaluation = self.evaluationFunction(state)
                    return evaluation
                else: #if max depth is not yet achieved, start another round
                    return expectimax(depth + 1, state, maxLayer)
            #other than min layer
            else:
                moves = state.getLegalActions(agentIndex)
                if len(moves) == 0: #in case there are no possible moves, re-evaluate
                    evaluation = self.evaluationFunction(state)
                    return evaluation
                #every possible consequent move is retrieved as the next layer of nodes
                nodes = (expectimax(depth, state.generateSuccessor(agentIndex, move), agentIndex + 1) for move in moves)

                #from the maximum layer the best option is returned
                if agentIndex == maxLayer:
                    return max(nodes)
                #return expectimax values if not in max layer
                else:
                    x = list(nodes)
                    return sum(x) / len(x)
                
        #Executing the expectimax, going through and sorting the results with their indices and selecting the one with the highest scores
        action = max(gameState.getLegalActions(0), key=lambda x: expectimax(1, gameState.generateSuccessor(0, x), 1))

        return action

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #retrieving data regarding game state
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newCapsules = currentGameState.getCapsules()
    
    #default values for variables
    food = newFood.asList()
    foodCount = - 10 * len(food)
    capsule = 0
    cap = 50
    distance_G = -600
    nearestFood = 0
    Gmod = -2

    #determining value for capsules
    if newCapsules:
        capsule = min([manhattanDistance(newPos, caps) for caps in newCapsules])
        cap = -3 / capsule

    #distance to ghost and the value it has
    ghost = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
    if ghost:
        distance_G = Gmod / ghost

    #distance to to food
    if food:
        nearestFood = min([manhattanDistance(newPos, f) for f in food])

    #total score
    return Gmod * nearestFood + distance_G + foodCount + cap

# Abbreviation
better = betterEvaluationFunction

