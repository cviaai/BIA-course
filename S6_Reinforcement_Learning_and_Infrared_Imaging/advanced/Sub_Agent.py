import numpy as np

class Sub_Agent:
    def __init__(self, name, names, actions, num_states, step_size, window_size):
        # Assumption that Still operation is 0
        self.actions     = actions
        self.num_actions = len(actions)+1
        self.num_states  = num_states
        self.window_size = window_size
        self.step_size   = step_size
        self.debug = False
        self.prev_state = None
        self.done = False
        self.q = None
        self.prev_node = None
        self.name = name
        self.names = names
        #print("Agent {} init".format(self.name))
        
        # generate empty Q-Window table
        lis = []
        for a in range(self.window_size):
            lis.append(self.num_states)
            lis.append(self.num_actions)

        self.q = np.ones(lis) # The array of action-value estimates.
        
        self.prev_node = None
        
    def agent_start(self, obs):
        #print("Agent {} start".format(self.name))
        """The first method called when the sub_agent starts
        Args:
            obs (int): the state observation from Master.
        """
        
        """Select first action from fake history of [obs, action_still,... , obs :] 
        the agent "think" that he didn't move and received the same reward
        """
  
        lis = []
        for i in range(self.window_size):
            lis.append(obs)
            lis.append(0) #Still
            
        self.prev_state = tuple(lis)
        
        self.done = False
        self.last_action = 0
        self.prev_node = None
        return 
    
    def agent_policy(self, state_action):
        """ policy of the agent
        Args:
            state_action (Numpy array)

        Returns:
            The action selected according to the policy
        """
                
        # compute softmax probability
        # Set the constant c by finding the maximum of state-action preferences
        c = np.max(state_action[1:])
            
        # Compute the numerator by subtracting c from state-action preferences and exponentiating it
        numerator = np.exp(state_action[1:] - c)
        
        # Next compute the denominator by summing the values in the numerator (use np.sum)
        denominator = numerator.sum()
             
        # Create a probability array by dividing each element in numerator array by denominator
        # We will store this probability array in self.softmax_prob as it will be useful later when updating the Actor
        
        softmax_prob = np.array([numerator[a]/denominator for a in range(len(numerator))])
        
        #print("Äctions: {}".format(self.actions))
        #print("probability: {}".format(softmax_prob))
        
        # Sample action from the softmax probability array
        # self.rand_generator.choice() selects an element from the array with the specified probability
        chosen_action = np.random.RandomState().choice(self.num_actions-1, p=softmax_prob)
        #while chosen_action == 0:
        #    chosen_action = np.random.RandomState().choice(self.num_actions, p=softmax_prob)

        # save softmax_prob as it will be useful later when updating the Actor
        self.softmax_prob = softmax_prob
        
        return chosen_action
        
        
    def agent_step(self, obs, reward, debug=False):
        """A step taken by the agent.
        Args:
            observation (int): the state observation from the
                environment's step based on where the agent ended up after the
                last step.
        Returns:
            action (int): the action the agent is taking.
        """
        
        if debug:
            print("Inside agent step")
            print("self.prev_state: {}".format(self.prev_state))
            

        try:
            # remove oldest pair of (obs, action) from history and attach current observation
            tail_obs=np.append(self.prev_state[2:], obs)
            current_q = self.q[tuple(tail_obs)]
        except:
            
            print("Step Obs: {}; Reward: {}".format(obs, reward))
            print("agent_step Last action: {}".format(self.prev_state[-1]))
            print("tail_obs: {}".format(tail_obs))
            
            
        if debug:
            print("Step Obs: {}; Reward: {}".format(obs, reward))
            print("agent_step Last action: {}".format(self.prev_state[-1]))        
            print("current_q: {}".format(current_q))
        
        # UPDATE STATE SPACE FOR TERMINATION
        if self.prev_node == None:   
            self.prev_node = Node(0, self.actions)
            if debug:
                print("First node: {}".format(self.prev_node))
                print(self.prev_node.printt())
        else:
            #
            if debug:
                print("Actions: {}".format(self.actions))
                print("Last action: {}".format(self.actions[self.last_action]))
                print("Previous node: {}".format(self.prev_node))
                print(self.prev_node.printt())
                print("Before node change, current node's Child? {}".format(self.prev_node.get_child(self.actions[self.last_action])))
            #if list(self.prev_node.get_child(action).keys())[0] == None:
            if self.prev_node.get_child(self.actions[self.last_action]) == None:
                if debug:
                    print("No child, creating")
                    print("Actions: {}".format(self.actions))
                node = Node(obs-1, self.actions)
            else:
                if debug:
                    print("Child exist: {}".format(self.prev_node.get_child(self.actions[self.last_action])))
                node = self.prev_node.get_child(self.actions[self.last_action])["node"]
            if debug:
                print("Current node: {}".format(node))
                print("Last action: {}".format(self.actions[self.last_action]))
            self.prev_node.next[self.last_action] = {"value": obs-1, "node": node}
            inverse_action = node.inverse_action(self.last_action)
            node.next[inverse_action] = {"value": (obs-1)*-1, "node": self.prev_node}
            #print("post")
            if debug:
                node.printt()
                if node.is_terminal():
                    print(node.is_terminal())
            self.prev_node = node
        
        self.done = self.prev_node.is_terminal()
        if self.done:
            if debug:
                print("find terminal state!!!!!!!!!!!!!")
            self.last_action = 0
        else:
            # action selection via Softmax
            self.last_action = self.agent_policy(current_q)
            if debug:
                print(self.names)
                print("action selected raw number: {}".format(self.last_action))
                print("action selected: {} {}".format(self.actions[self.last_action], self.names[self.last_action]))

            # Perform update
            # --------------------------
            delta = self.step_size * reward
            if debug:
                print("delta: {}".format(delta))
                print("Q values before update: {}".format(self.q[tuple(self.prev_state[:-1])]))

            # subtract delta from all the actions values of the current state 
            self.q[tuple(self.prev_state[:-1])] -=  delta

            if debug:          
                print("self.q after subtract: {}".format(self.q[tuple(self.prev_state[:-1])]))


            # add 2*delta to selected action value of the current state (compensate previous subtract)
            self.q[tuple(self.prev_state)] += 2*delta

            if debug:          
                print("2*delta: {}".format(2*delta))
                print("self.q updated: {}".format(self.q[tuple(self.prev_state[:-1])]))

            # limit values in range [-20, +20]           
            temp = self.q[tuple(self.prev_state[:-1])]
            temp[temp<-50]=-50
            temp[temp>50]=50
            self.q[tuple(self.prev_state[:-1])] = temp

            if debug:          
                print("self.q updated: {}".format(self.q[tuple(self.prev_state[:-1])]))
            # --------------------------            

            self.prev_state =  np.append(tail_obs, self.last_action+1)
    
 
        return self.actions[self.last_action]
      
    def ag_print(self):
        print(self.actions)
        print(self.num_actions)
        print(self.num_states)
        print(self.max_step)
        
        
class Node:
        
    def __init__(self, data = None, actions = [0]):
        self.data = data
        self.actions = actions
        self.next = { x: None for x in actions}
    
    def printt(self):
        print("Value: {}".format(self.data))
        #print("Actions: {}".format(self.actions))
        print("Next: {}".format(self.next))

    def inverse_action(self, action):
        #print("Actions : {}".format(self.actions))        
        #print("Action received: {}".format(action))
        #print("Action adjusted: {}".format(self.actions[action-1]))
        ret = self.actions.copy()
        ret.remove(self.actions[action])
        
        pop = ret.pop()
        #print("Action inverse: {}".format(pop))
        return pop
    
    def get_child(self, action):
        ret = None
        child = self.next.get(action)
        if  child != None:
            ret = child
        return ret
    
    
    def is_explored(self):
#         print("self.next: {}" .format(self.next))
#         print("answer: {}".format(None in self.next.values()==False))
        return not (None in self.next.values())

    def is_terminal(self):
        ret = False
        
        if self.data == 2:
            ret = True
        elif self.is_explored():
            ret = True
            #print("Self data: {}".format(self.data))
            for i in self.actions:
                #print(self.get_child(i)["value"])
                if self.get_child(i)["value"] > self.data:
                    ret = False
        else:
            ret = True
            counter = 0
            for i in self.actions:
                child = self.get_child(i)
                if child != None:
                    counter +=1
                    if child["value"] > self.data:
                        ret = False  
            if counter == 0:
                ret = False
        return ret        