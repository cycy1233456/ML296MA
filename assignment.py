import dataclasses
import math
import typing

#######################################################################################
# PART ONE
#   - You only need to implement the six "load" functions below.
#   - On Gradescope there will be two submissions:
#       1. Implement the six "load" functions and turn in this file.
#       2. Submit proofs that your test cases and expected results are correct.
#   - Do not change the file name or any function signatures.
#######################################################################################


@dataclasses.dataclass(frozen=True)
class Q1TestCase:
    pi_0: dict[int, float]
    n: int
    expected_result: dict[int, float]


@dataclasses.dataclass(frozen=True)
class Q2TestCase:
    n: int
    T: int
    S_0: int
    c: float
    expected_result: float


@dataclasses.dataclass(frozen=True)
class Q3TestCase:
    N: int
    alpha: float
    beta: float
    expected_result: float


def load_question_1_test_case_example() -> Q1TestCase:
    """Nothing for you to do here; this is what a test case should look like."""
    pi_0 = {2: 1.0}
    n = 2
    pi_n = {0: 1 / 3, 1: 5 / 24, 2: 5 / 24, 4: 1 / 4}
    return Q1TestCase(pi_0=pi_0, n=n, expected_result=pi_n)


def load_question_1_test_case_1() -> Q1TestCase:
    raise NotImplementedError


def load_question_1_test_case_2() -> Q1TestCase:
    raise NotImplementedError


def load_question_2_test_case_1() -> Q2TestCase:
    raise NotImplementedError


def load_question_2_test_case_2() -> Q2TestCase:
    raise NotImplementedError


def load_question_3_test_case_1() -> Q3TestCase:
    raise NotImplementedError


def load_question_3_test_case_2() -> Q3TestCase:
    raise NotImplementedError


#######################################################################################
# PART TWO
#   - You must implement the following three "solve" functions.
#   - The autograder will run them against a test suite.
#   - Do not change any of the function signatures.
#######################################################################################


def solve_question_1(pi_0: dict[int, float], n: int) -> dict[int, float]:
    """Markov chain on infinite state space.

    Args:
        pi_0: Dictionary mapping states to probabilities. You may assume all keys are
            non-negative integers, all values are non-negative, and the values sum
            to 1.
        n: Positive integer; the number of timesteps to simulate.

    Returns:
        A dictionary mapping states to probabilities. Any state not in the dictionary
        is assumed to have probability zero.
    """
    current_probabilities = pi_0.copy()

    for _ in range(n):
        print(f"Calculating {_+1}th iteration Pr(X_{_+1}=x)")
        new_probabilities={}
        # if the largest state in current state distribution is n
        # after the transition, the largest should be no more than n+1
        # we should calculate P(s') = sum(s)[P(S_t+1=s'|S_t=s)*pi(s)]
        for i in range(max(current_probabilities.keys())+2):
            if i == 0:
                state_prob = current_probabilities[0] * 0.5
            else:
                state_prob = 0
            for j in range(1, max(current_probabilities.keys())+1):
                state_prob += (1/(2*j)) * current_probabilities[j]
            new_probabilities[i] = state_prob
        current_probabilities = new_probabilities

    return current_probabilities


def solve_question_2(n: int, T: int, S_0: int, c: float) -> float:
    """Finite-horizon Markov decision process.

    Args:
        n: Positive integer; the number of shares you must sell.
        T: Positive integer; finite time horizon.
        S_0: Positive integer; the initial stock price. You may assume S_0 > T + 1.
        c: Positive float; penalty parameter for over-selling.

    Returns:
        Total expected revenue under an optimal policy.
    """
    # construct state_value_tableau SV[i,j]=V(s=i,t=j) left i stocks at time j
    # time is from 0 to T (total T+1) and stocks num is from 0 to 2n(total 2n+1)
    State_Value = [[0]*(T+1) for i in range(2*n+1)]
    Optimal_Policy = [[0]*(T+1) for i in range(2*n+1)]
    Stock_Price = [(S_0 + day) for day in range(T+1)]

    # define the state value at T
    for m_T in range(2*n+1):
        # last column (at time T, all the stock must be sold)
        State_Value[m_T][T] = 0.5 * m_T * Stock_Price[T]

    # backward induction from T-1
    # use the policy evaluation to find the pi*
    for t in range(T-1, -1, -1):
        # print(t)
        for m_t in range(2*n+1):
            # find the state value of (m_t, t)
            # we can choose the action: no more than current left stock m_t
            feasible_actions = [a_t for a_t in range(m_t+1)]
            feasible_action_values = []
            best_action = 0
            best_action_value = -1e10
            for a_t in feasible_actions:
                # given a_t to calculate the estimate action value
                # we should find the probability of K_t>= at
                # since K~Binomial(n/2, 1/2) if K < a_t then i can choose {0,1,2...a_t - 1}
                if a_t > n/2:
                    # K_max = n/2 so K must be smaller than a_t
                    P_K_larger = 0
                    P_K_smaller = 1
                elif a_t == 0:
                    P_K_larger = 1
                    P_K_smaller = 0
                else:
                    # example if n=6 n/2=3 and a_t=3 then range(a_t)=[0,1] so P_K_smaller=0.5
                    P_K_smaller = sum([n_choose_k(int(n/2), int(i))*(0.5**(n/2)) for i in range(a_t)])
                    P_K_larger = 1 - P_K_smaller

                # if K >= a_t sell a_t at price S_t
                stock_price = Stock_Price[t]
                current_return1 = stock_price * a_t
                next_state_value1 = State_Value[m_t - a_t][t+1]
                action_value1 = P_K_larger * (current_return1 + next_state_value1)

                # if K < a_t then sell K_t we should find the probability of K to find the p(s'|m_t, t, a_t)
                # calculate E[q(m_t, t, a_t)|a_t>K_t]
                action_value2 = 0
                for K_t in range(a_t):
                    # excess sell should pay more
                    prob_k = n_choose_k(int(n/2), int(K_t))*(0.5**(n/2))
                    current_return2 = stock_price * K_t - c*(a_t - K_t)
                    next_state_value2 = State_Value[m_t - K_t][t+1]
                    action_value2 += prob_k * (current_return2 + next_state_value2)

                # the total expected q(m_t, t, a_t, K_t)
                current_action_value = action_value1 + action_value2
                # update the best action based on the policy evaluation
                feasible_action_values.append(action_value1 + action_value2)
                if current_action_value > best_action_value:
                    best_action_value = current_action_value
                    best_action = a_t
            # update the state_value_tableau and policy tableau
            State_Value[m_t][t] = best_action_value
            Optimal_Policy[m_t][t] = best_action

    # return State_Value, Optimal_Policy
    return State_Value[2*n][0]



def solve_question_3(N: int, alpha: float, beta: float) -> float:
    """Grid world Markov decision process

    Args:
        N: Positive odd integer; size of the grid.
        alpha: Non-negative float; penalty parameter.
        beta: Non-negative float; penalty parameter.

    Returns:
        Total expected reward under an optimal policy.
    """
    import copy
    # VALUE ITERATION
    # discount factor
    Discount_Factor = 0.5
    # DEFINE THE ACTION SET A
    ACTIONS_SET = ['U', 'R', 'D', 'L']
    # CALCULATE THE ACTIONS VALUE FOR DIFFERENT ACTION IN CURRENT STATE
    
    
    def MAXIMUM_NORM(table1, table2):
        flat_table1 = [elem for row in table1 for elem in row]
        flat_table2 = [elem for row in table2 for elem in row]
        return max([abs(flat_table1[i] - flat_table2[i]) for i in range(len(flat_V_table_k1))])
        

    # there are three tables Q-table is the optimal action-value
    # V-table is the state value R table is the current return
    # initial Qtable all 0
    Q_Table = [[0]*N for _ in range(N)]
    # initial Vtable is all 0
    V_Table = [[0]*N for _ in range(N)]
    R_Table = [[0]*N for _ in range(N)]

    # SET THE CURRENT RETURN
    R_Table[N-1][N-1] = 1
    R_Table[(N - 1) // 2][(N - 1) // 2] = -alpha
    R_Table[N-1][0] = -beta

    count = 0
    V_Table_k1 = copy.deepcopy(V_Table)
    
    # setting the iteration end condition: when state value converge (V_k+1 - V_k) use maximum norm   
    while (count == 0) or (MAXIMUM_NORM(V_Table, V_Table_k1)> 0.00001):
        
        # use the new state value table
        V_Table = copy.deepcopy(V_Table_k1)
        count += 1
        print(count)
        # VALUE ITERATION BEGIN
        # V_k is restored in V_TABLE update from VTABLE
        # every iteration update the Q_table to find the max action_value
        for i in range(N):
            for j in range(N):
                # THIS DICT CALCULATE THE FOUR ACTION AT state s find q(s,a)
                # CURRENT_STATE_ACTIONS_VALUES = dict(zip(ACTIONS_SET, [0] * 4))
                for action in ACTIONS_SET:
                    """ For each action we first find the after_action_position and then consider the blow
                        After we affirm the adj_position(adj_i, adj_j) we only need to find the value of 
                        Blow Left: (adj_i, adj_j-1)  Blow Down: (adj_i+1, adj_j)
                    """
                    if action == 'U':
                        # we consider the condition that our action will heat the boundary if A = U or L the index will decrease
                        # use max(i-k,0) to define the boundary case
                        # if at the upper boundary, the action 'U' equals to No Action, only based on the blow
                        if i == 0:
                            after_action_i = i  # meet boundary directly come back == No move
                            after_action_j = j
                        else:
                            after_action_i = i - 1
                            after_action_j = j
                        # p=0.5 stay at (adj_i, adj_j) p=0.25 blow L p=0.25 blow D
                        estimate_action_value = (0.5 * V_Table[after_action_i][after_action_j]
                                                + 0.25 * V_Table[min(after_action_i+1, N-1)][after_action_j]  # DOWN
                                                 + 0.25 * V_Table[after_action_i][max(after_action_j-1, 0)])  # LEFT
                        # q(s,a) = sum(r)pi(r|a,s) * r + sum(s')p(s'|s,a)*V_k(s')
                        CURRENT_STATE_ACTIONS_VALUES[action] = Discount_Factor * estimate_action_value + R_Table[i][j]
                        
                    elif action == 'R':
                        if j == N-1:
                            after_action_i = i
                            after_action_j = j
                        else:
                            after_action_i = i
                            after_action_j = j + 1
                        # we consider the condition that our action will heat the boundary if A = D or R the index will increase
                        estimate_action_value = (0.5 * V_Table[after_action_i][after_action_j]
                                                 + 0.25 * V_Table[min(after_action_i + 1, N - 1)][after_action_j]  # DOWN
                                                 + 0.25 * V_Table[after_action_i][max(after_action_j - 1, 0)])  # LEFT
                        CURRENT_STATE_ACTIONS_VALUES[action] = Discount_Factor * estimate_action_value + R_Table[i][j]
                        
                    elif action == 'D':
                        if i == N-1:
                            after_action_i = i
                            after_action_j = j
                        else:
                            after_action_i = i + 1
                            after_action_j = j
                        estimate_action_value = (0.5 * V_Table[after_action_i][after_action_j]
                                                 + 0.25 * V_Table[min(after_action_i + 1, N - 1)][after_action_j]  # DOWN
                                                 + 0.25 * V_Table[after_action_i][max(after_action_j - 1, 0)])  # LEFT
                        CURRENT_STATE_ACTIONS_VALUES[action] = Discount_Factor * estimate_action_value + R_Table[i][j]
                        
                    elif action == 'L':
                        if j == 0:
                            after_action_i = i
                            after_action_j = j
                        else:
                            after_action_i = i
                            after_action_j = j - 1
                        estimate_action_value = (0.5 * V_Table[after_action_i][max(after_action_j-1, 0)]
                                                 + 0.25 * V_Table[min(after_action_i + 1, N - 1)][after_action_j]  # DOWN
                                                 + 0.25 * V_Table[after_action_i][max(after_action_j - 1, 0)])  # LEFT
                        CURRENT_STATE_ACTIONS_VALUES[action] = Discount_Factor * estimate_action_value + R_Table[i][j]
                
                # choose the largest as the action_value --> pi(a*|s) = 1
                best_action_value = max(CURRENT_STATE_ACTIONS_VALUES.values())
                Q_Table[i][j] = best_action_value

        # update the state value
        V_Table_k1 = copy.deepcopy(Q_Table)

    return V_Table_k1



#######################################################################################
# UTILITY FUNCTIONS
#######################################################################################


def init_matrix(
    *, nrows: int, ncols: int, fill_value: typing.Any = 0
) -> list[list[int]]:
    return [[fill_value] * ncols for _ in range(nrows)]


def n_choose_k(n: int, k: int) -> int:
    return math.comb(n, k)
