import numpy as np
import os
import Sub_Agent as sub_agent


class Master_Agent(object):
    def __init__(self):
            self.window_size   = None 
            self.num_actions   = 5
            self.num_states    = None            
            self.still         = 0
            self.actions       = None
            self.step_size     = 1.0
            self.q_path        = ""
            self.learning      = True
            self.agent_size    = None
            self.debug         = False
            self.prev_rew      = None
            self.prev_true_obs = None 
            self.sub_agents    = None
            self.subagent_step    = 0
            self.max_step      = 50
            self.prev_super    = None
            self.last_action = None
            
            
            self.macro_names = ["Still", "Vertical", "Horizontal", "Rotation", "Scaling"]
            self.names = ["Still","up", "down","left", "right","clockwise", "counterclock","increase", "decrease"]
            
    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts.

        Args:
        agent_init_info (dict), the parameters used to initialize the agent. The dictionary contains:
        {
            window_size (int)            : q window size of (obs, action) history
            num_states (int)             : The number of states
            num_actions (int)            : The number of actions
            step_size (float)            : The step-size for updating q values
            still (int)                  : number representing the no-op action
            actions (list of list of int): antagonist actions for the sub agents
            agent_size (int)             : q window size of (obs, action) history for the sub agents
            path (string)                : optional, path to saved model
            learning (bool)              : flag indicating if the system is updating the q table or not
            debug (bool)                 : optional, flag for debugging prints
        }

        """
        # Store the parameters provided in agent_init_info.   


        self.window_size = agent_info["window_size"]  
        
        if "num_actions" in agent_info:
             self.num_actions = agent_info["num_actions"]
                
        self.num_states = agent_info["num_states"]

        if "still" in agent_info:
            self.still = agent_info["still"]

        self.actions = agent_info["actions"]

        if "step_size" in agent_info:
            self.step_size = agent_info["step_size"]            

        self.agent_size = agent_info["agent_size"]              
            
        if "path" in agent_info:
            self.q_path = agent_info["path"]
        if os.path.isfile(self.q_path):
            self.q = np.load(self.q_path)
        else:
            # Create an array for action-value estimates and initialize it to zero.
            lis = []
            for a in range(self.window_size):
                lis.append(self.num_states)
                lis.append(self.num_actions)
            self.q = np.ones((lis), dtype='float32')

        if "debug" in agent_info:
            self.debug = agent_info["debug"]

        if "learning" in agent_info:
            self.learning = agent_info["learning"]   

        # sub agents initialization                                    #actions             num_states       step_size       window_size       
        self.sub_agents = [[],  
            sub_agent.Sub_Agent("vertical", self.names[1:3],    self.actions["vertical"],   self.num_states, self.step_size, 3),
            sub_agent.Sub_Agent("horizontal", self.names[3:5],  self.actions["horizontal"], self.num_states, self.step_size, 3),
            sub_agent.Sub_Agent("rotation", self.names[5:7],    self.actions["rotation"],   self.num_states, self.step_size, 3),
            sub_agent.Sub_Agent("scaling", self.names[7:],      self.actions["scaling"],    self.num_states, self.step_size, 3)
                     ] 
        
         
    def agent_start(self, obs):
        """The first method called when the sub_agent starts
        Args:
            obs (int): the state observation from Environment.
        """
        
        """Select first action from fake history of [obs, action_still,... , obs :] 
        the agent think that he didn't move and received the same reward
        """
 
        # initial state with fake history of Same observation and still actions
        adj_obs, _ = self.process_observation(obs, obs, 0)
        lis = []
        for i in range(self.window_size):
            lis.append(adj_obs)
            lis.append(self.still)
            
        self.prev_state = tuple(lis)
        
        self.last_action = self.still
        
        self.subagent_step = 0
        self.prev_super = None
        self.prev_true_obs = None
        self.last_action = self.still
    
    
    # Master agent action selection function
    def agent_step(self, obs, reward, debug=False):
        action = self.step_inner(obs, reward, debug)
        if self.last_action != self.still:
            if action == self.still and self.sub_agents[self.last_action].done == True:
                action = self.step_inner(obs, reward, debug)
        return action
        
    
    def step_inner(self, obs, reward, debug=False):
        """A step taken by the agent.
        Args:
            observation (int): the state observation from the
                environment's step based on where the agent ended up after the
                last step.
                
        Returns:
            action (int): the action the agent is taking.
        """  
        
        action_ret = None
        
        change = False
        
        # time to select new action/sub agent
        if self.last_action == self.still or self.sub_agents[self.last_action].done == True or self.subagent_step > self.max_step:      

            # cost is sub agent steps (on optimal Still better than moving around)
            cost = 0
            if self.last_action != self.still:
                cost = self.subagent_step
                self.subagent_step=0

            
            adj_obs, rew = self.process_observation(obs, self.prev_super, cost)
            if debug:
                print("New master action: original obs {}, previous {} adjusted {}, reward {}".format(obs, self.prev_super, adj_obs, rew))
            self.prev_super = obs
                   
            
            # remove oldest pair of (obs, action) from history and attach current observation
            tail_obs=np.append(self.prev_state[2:], adj_obs)
            current_q = self.q[tuple(tail_obs)]

            if debug:
                print("Step Obs: {}; Reward: {}".format(adj_obs, rew))
                print("agent_step Last action: {}".format(self.prev_state[-1]))
                print("time to change")
            change = True
            
            # action selection via Softmax
            self.last_action = self.agent_policy(current_q)

            if debug:
                print("New agent selected: " + self.macro_names[self.last_action])

            # Perform update
            # --------------------------
            delta = self.step_size * rew
            if self.debug:
                print("delta: {}".format(delta))
                print("Q values before update: {}".format(self.q[tuple(self.prev_state[:-1])]))

            # subtract delta from all the actions values of the current state 
            self.q[tuple(self.prev_state[:-1])] -=  delta

            if self.debug:          
                print("self.q after subtract: {}".format(self.q[tuple(self.prev_state[:-1])]))


            # add 2*delta to selected action value of the current state (compensate previous subtract)
            self.q[tuple(self.prev_state)] += 2*delta

            if self.debug:          
                print("2*delta: {}".format(2*delta))
                print("self.q updated: {}".format(self.q[tuple(self.prev_state[:-1])]))

            # limit values in range [-50, +50]           
            temp = self.q[tuple(self.prev_state[:-1])]
            temp[temp<-50]=-50
            temp[temp>50]=50
            self.q[tuple(self.prev_state[:-1])] = temp

            if self.debug:          
                print("self.q updated: {}".format(self.q[tuple(self.prev_state[:-1])]))
            # --------------------------

            self.prev_state =  np.append(tail_obs, self.last_action)
        
        # go with sub agent
        if self.last_action != self.still: 
            #sub agent start
            if change:
                if debug:
                    print("agent step 0")
                self.prev_true_obs = None
                adj_obs, rew = self.process_observation_agent(obs, self.prev_true_obs)
                self.prev_true_obs = obs
                self.sub_agents[self.last_action].agent_start(adj_obs)    
            else:
                # sub agent already started, process observation
                adj_obs, rew = self.process_observation_agent(obs, self.prev_true_obs)
                
                if debug:
                    print("continue: original obs {}, previous {} adjusted {}, reward {}".format(obs, self.prev_true_obs, adj_obs, rew))
                self.prev_true_obs = obs    
                    
            action_ret = self.sub_agents[self.last_action].agent_step(adj_obs, rew, debug)
            # if sub agent end force a still action
            if self.sub_agents[self.last_action].done:
                action_ret = self.still
                
            self.subagent_step += 1        
        else:
            #print("Still action")
            action_ret = self.last_action
        if debug:
            print("Action from master: {}".format(self.names[action_ret]))
        return action_ret    
                      
    def agent_policy(self, state_action, debug=False):
        """ policy of the agent
        Args:
            state_action (Numpy array): 

        Returns:
            The action selected according to the policy
        """
        if debug:
            print("\n\nagent_policy")
            print("Initial state_action vector: {}".format(state_action))

        # compute softmax probability
        max_q = np.max(state_action)

        if debug:
            print("Max action value: {}".format(max_q))  


        # Compute the numerator by subtracting c from state-action preferences and exponentiating it
        numerator = np.exp(state_action - max_q)
        #input_max, input_indexes

        # Next compute the denominator by summing the values in the numerator 
        denominator = numerator.sum()

        # Create a probability array by dividing each element in numerator array by denominator
        softmax_prob = np.array([numerator[a]/denominator for a in range(len(numerator))])

        if debug:
            print("Softmax probability vector: {}".format(softmax_prob))

        # Sample action from the softmax probability array
        chosen_action = np.random.RandomState().choice(self.num_actions, p=softmax_prob)

        return chosen_action                       
    
    def process_observation_agent(self, observation, prev_obs):
            """Helper function called for translate the observation received into a smaller set of
            values used for state representation and reward
            Args:
                observation (int): the state observation from the environment (range [0, 1000]).
                prev_obs (int):    old state observation from the environment (range [0, 1000]).
            Returns:
                obs (int): step reward (range [-1, 2]).
            """

            """ If this function is called in env.reset when there is not yet a previous observation
            so here we set it as current one"""
            #print("process_observation self.prev_true_obs: " + str(self.prev_true_obs))
            if prev_obs == None:
                prev_obs = observation

            #print("process_observation prev_obs: {}".format(self.prev_true_obs))

            """- If the observation received is =1000 it means the agent is in the correct position, thus 2 reward
            - If the observation received is > than the previous it means the agent moved towards the 
            correct position, assign 1 reward
            - If the observation received is = to the previous it means the agent made a useless  move, 
            we want to discourage it, 0 reward
            - If the observation received is < than the previous it means the agent moved away from the direction
            leading to the correct position, -1 reward""" 

            obs, rew = 0,0

            #print(observation, self.prev_true_obs)
            if observation == 1000:
                rew = 2  
                obs = 3
            elif observation > prev_obs:
                rew = 1
                obs = 2
            elif observation == prev_obs:
                rew = 0
                obs = 1
            else:        
                rew = -1
                obs = 0

            return [obs, rew]              
                      
    
    def process_observation(self, observation, prev_obs, cost):
            """Helper function called for translate the observation received into a smaller set of
            values used for state representation and reward
            Args:
                observation (int): the state observation from the environment (range [0, 1000]).
                prev_obs (int):    old state observation from the environment (range [0, 1000]).
            Returns:
                obs (int): step reward (range [-1, 2]).
            """

            """ If this function is called in env.reset when there is not yet a previous observation
            so here we set it as current one"""
            #print("process_observation self.prev_true_obs: " + str(self.prev_true_obs))
            if prev_obs == None:
                prev_obs = observation

            #print("process_observation prev_obs: {}".format(self.prev_true_obs))

            """- If the observation received is =1000 it means the agent is in the correct position, thus 2 reward
            - If the observation received is > than the previous it means the agent moved towards the 
            correct position, assign 1 reward
            - If the observation received is = to the previous it means the agent made a useless  move, 
            we want to discourage it, 0 reward
            - If the observation received is < than the previous it means the agent moved away from the direction
            leading to the correct position, -1 reward""" 

            obs, rew = 0,0

            #print(observation, self.prev_true_obs)
            if observation == 1000 and cost==0:
                rew = 2  
                obs = 3
            elif observation > prev_obs:
                rew = 1
                obs = 2
            elif observation == prev_obs:
                rew = -1
                obs = 1
            else:        
                rew = -2
                obs = 0

            return [obs, rew] 