import matplotlib.pyplot as plt
import random
import gymnasium as gym
import requests
import warnings
import logging
import numpy as np
import pandas as pd
import inspect
import json
import os

from collections import OrderedDict
from scipy import interpolate
from pprint import pformat
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback

from .examples.test_and_plot import plot_results, test_agent


#==============================================================================
# --- VERIFY these names match your specific BOPTEST test case observations/actions ---
ZONE_TEMP_OBS_NAME = 'reaTZon_y'        # Zone temperature observation name

# --- Use Heating Setpoint Measurement as Occupancy Proxy ---
OCCUPANCY_PROXY_OBS_NAME = 'reaTSetHea_y' # Verified measurement name

# --- Define expected setpoint values (in Kelvin if obs are K) ---
# These should match the baseline controller's logic as observed or documented
OCCUPIED_HEATING_SETPOINT_K = 273.15 + 21.0  # Assumed occupied heating SP (21 C)
UNOCCUPIED_HEATING_SETPOINT_K = 273.15 + 15.0 # Assumed unoccupied heating SP (15 C)
# Tolerance for float comparison when checking the proxy
SETPOINT_COMPARISON_TOLERANCE = 0.1 # Kelvin tolerance

# --- Define the heating action name ---
HEATING_ACTION_NAME = 'oveHeaPumY_u'    # Verified heating action name

# --- Define Setpoints and Margin for the MASKING RULE ITSELF (in Celsius) ---
OCCUPIED_UPPER_SP_C = 24.0
OCCUPIED_LOWER_SP_C = 21.0
UNOCCUPIED_UPPER_SP_C = 30.0 # Example: Allow wider range when unoccupied
UNOCCUPIED_LOWER_SP_C = 15.0 # Example: Allow wider range when unoccupied
SAFETY_MARGIN_C = 1.0      # +/- 1 degree Celsius buffer

# --- Unit Conversion and Assumption ---
ASSUME_OBS_IN_KELVIN = True # Set to False if your specific obs are Celsius

def celsius_to_kelvin(temp_c):
    return temp_c + 273.15

if ASSUME_OBS_IN_KELVIN:
    # Convert masking rule setpoints to Kelvin
    OCCUPIED_UPPER_SP_K = celsius_to_kelvin(OCCUPIED_UPPER_SP_C)
    OCCUPIED_LOWER_SP_K = celsius_to_kelvin(OCCUPIED_LOWER_SP_C)
    UNOCCUPIED_UPPER_SP_K = celsius_to_kelvin(UNOCCUPIED_UPPER_SP_C)
    UNOCCUPIED_LOWER_SP_K = celsius_to_kelvin(UNOCCUPIED_LOWER_SP_C)
    SAFETY_MARGIN_K = SAFETY_MARGIN_C # Margin is same magnitude in K or C
    # Keep proxy check setpoints in K as defined above
else:
    # Use Celsius directly for masking rule setpoints
    OCCUPIED_UPPER_SP_K = OCCUPIED_UPPER_SP_C
    OCCUPIED_LOWER_SP_K = OCCUPIED_LOWER_SP_C
    UNOCCUPIED_UPPER_SP_K = UNOCCUPIED_UPPER_SP_C
    UNOCCUPIED_LOWER_SP_K = UNOCCUPIED_LOWER_SP_C
    SAFETY_MARGIN_K = SAFETY_MARGIN_C
    # Need to ensure proxy check values are also in Celsius if obs aren't Kelvin
    if OCCUPIED_HEATING_SETPOINT_K > 100: # Basic check if K values were used
         warnings.warn("ASSUME_OBS_IN_KELVIN is False, but proxy setpoints seem to be in Kelvin. Adjust constants.", RuntimeWarning)

class BoptestGymEnv(gym.Env):
    '''
    BOPTEST Environment that follows gym interface.
    This environment allows the interaction of RL agents with building
    emulator models from BOPTEST. 
     
    '''
    
    metadata = {'render.modes': ['console']}

    def __init__(self, 
                 url                = 'http://127.0.0.1',
                 testcase           = 'bestest_hydronic_heat_pump',
                 actions            = ['oveHeaPumY_u'],
                 observations       = {'reaTZon_y':(280.,310.)}, 
                 reward             = ['reward'],
                 max_episode_length = 3*3600,
                 random_start_time  = False,
                 excluding_periods  = None,
                 regressive_period  = None,
                 predictive_period  = None,
                 start_time         = 0,
                 warmup_period      = 0,
                 scenario           = {'electricity_price':'constant'},
                 step_period        = 900,
                 render_episodes    = False,
                 log_dir            = os.getcwd()):
        '''
        Parameters
        ----------
        url: string
            Rest API url for communication with the BOPTEST interface
        testcase: string
            The string identifier of the testcase
        actions: list
            List of strings indicating the action space. The bounds of 
            each variable from the action space the are retrieved from 
            the overwrite block attributes of the BOPTEST test case
        observations: dictionary
            Dictionary mapping observation keys to a tuple with the lower
            and upper bound of each observation. Observation keys must 
            belong either to the set of measurements or to the set of 
            forecasting variables of the BOPTEST test case. Contrary to 
            the actions, the expected minimum and maximum values of the 
            measurement and forecasting variables are not provided from 
            the BOPTEST framework, although they are still relevant here 
            e.g. for normalization or discretization. Therefore, these 
            bounds need to be provided by the user. 
            If `time` is included as an observation, the time in seconds
            will be passed to the agent. This is the remainder time from 
            the beginning of the episode and for periods of the length
            specified in the upper bound of the time feature. 
        reward: list
            List with string indicating the reward column name in a replay
            buffer of data in case the algorithm is going to use pretraining
        max_episode_length: integer
            Maximum duration of each episode in seconds
        random_start_time: boolean
            Set to True if desired to use a random start time for each episode
        excluding_periods: list of tuples
            List where each element is a tuple indicating the start and 
            end time of the periods that should not overlap with any 
            episode used for training. Example:
            excluding_periods = [(31*24*3600,  31*24*3600+14*24*3600),
                                (304*24*3600, 304*24*3600+14*24*3600)]
            This is only used when `random_start_time=True`
        regressive_period: integer, default is None
            Number of seconds for the regressive horizon. The observations
            will be extended for each of the measurement variables indicated
            in the `observations` dictionary argument. Specifically, a number 
            of `int(self.regressive_period/self.step_period)` observations per
            measurement variable will be included in the observation space.
            Each of these observations correspond to the past observation 
            of the measurement variable `j` steps ago. This is used in partially
            observable MDPs to compensate for the hidden states. 
            Note that it is NOT allowed to use `regressive_period=0` since that
            would represent a case where you want to include a measurement at
            the current time in the observation space, which is directly done
            when adding such measurement to the `observations` argument. 
        predictive_period: integer, default is None
            Number of seconds for the prediction horizon. The observations
            will be extended for each of the predictive variables indicated
            in the `observations` dictionary argument. Specifically, a number
            of `int(self.predictive_period/self.step_period)` observations per 
            predictive variable will be included in the observation space.
            Each of these observations correspond to the foresighted 
            variable `i` steps ahead from the actual observation time. 
            Note that it's allowed to use `predictive_period=0` when the
            intention is to retrieve boundary condition data at the actual
            observation time, useful e.g. for temperature setpoints or 
            ambient temperature. 
        start_time: integer
            Initial fixed episode time in seconds from beginning of the 
            year for each episode. Use in combination with 
            `random_start_time=False` 
        warmup_period: integer
            Desired simulation period to initialize each episode 
        scenario: dictionary
            Defines the BOPTEST scenario. Can be `constant`, `dynamic` or
            `highly_dynamic`
        step_period: integer
            Sampling time in seconds
        render_episodes: boolean
            True to render every episode
        log_dir: string    
            Directory to store results like plots or KPIs
            
        '''
        
        super(BoptestGymEnv, self).__init__()
        
        self.url                = url
        self.testcase           = testcase
        self.actions            = actions
        self.observations       = list(observations.keys())
        self.max_episode_length = max_episode_length
        self.random_start_time  = random_start_time
        self.excluding_periods  = excluding_periods
        self.start_time         = start_time
        self.warmup_period      = warmup_period
        self.reward             = reward
        self.predictive_period  = predictive_period
        self.regressive_period  = regressive_period
        self.step_period        = step_period
        self.scenario           = scenario
        self.render_episodes    = render_episodes
        self.log_dir            = log_dir
        self.last_raw_observation = None
        self._obs_name_to_index = {}
        
        # Avoid requesting data before the beginning of the year
        if self.regressive_period is not None:
            self.bgn_year_margin = self.regressive_period
        else:
            self.bgn_year_margin = 0
        # Avoid surpassing the end of the year during an episode
        self.end_year_margin = self.max_episode_length
        
        #=============================================================
        # Get test information
        #=============================================================
        # Get testid for the particular testcase
        # Check if already started a test case and stop it if so before starting another
        try:
            requests.put('{0}/stop/{1}'.format(url, self.testid))
        except:
            pass
        # Select and start a new test case
        self.testid = requests.post('{0}/testcases/{1}/select'.format(url, testcase)).json()['testid']
        # Test case name
        self.name = requests.get('{0}/name/{1}'.format(url, self.testid)).json()['payload']
        # Measurements available
        self.all_measurement_vars = requests.get('{0}/measurements/{1}'.format(url, self.testid)).json()['payload']
        # Predictive variables available
        self.all_predictive_vars = requests.get('{0}/forecast_points/{1}'.format(url, self.testid)).json()['payload']
        # Inputs available
        self.all_input_vars = requests.get('{0}/inputs/{1}'.format(url, self.testid)).json()['payload']
        # Default simulation step
        self.step_def = requests.get('{0}/step/{1}'.format(url, self.testid)).json()['payload']
        # Default scenario
        self.scenario_def = requests.get('{0}/scenario/{1}'.format(url, self.testid)).json()['payload']
        
        #=============================================================
        # Define observation space
        #=============================================================
        # Assert size of tuples associated to observations
        for obs in self.observations:
            if len(observations[obs])!=2: 
                raise ValueError(\
                     'Values of the observation dictionary must be tuples '\
                     'of dimension 2 indicating the expected lower and '\
                     'upper bounds of each variable. '\
                     'Variable "{}" does not follow this format. '.format(obs))
        
        # Assert that observations belong either to measurements or to predictive variables
        for obs in self.observations:
            if not (obs=='time' or obs in self.all_measurement_vars.keys() or obs in self.all_predictive_vars.keys()):
                raise ReferenceError(\
                 '"{0}" does not belong to neither the set of '\
                 'test case measurements nor to the set of '\
                 'forecasted variables. \n'\
                 'Set of measurements: \n{1}\n'\
                 'Set of forecasting variables: \n{2}'.format(obs, 
                                                              list(self.all_measurement_vars.keys()), 
                                                              list(self.all_predictive_vars.keys()) ))
        
        # observations = measurements + predictions
        self.measurement_vars = [obs for obs in self.observations if (obs in self.all_measurement_vars)]
        
        # Initialize observations and bounds
        self.observations = []
        self.lower_obs_bounds = []
        self.upper_obs_bounds = []
        
        # Check for time in observations
        if 'time' in list(observations.keys()):
            self.observations.extend(['time'])
            self.lower_obs_bounds.extend([observations['time'][0]])
            self.upper_obs_bounds.extend([observations['time'][1]])
        
        # Define lower and upper bounds for observations. Always start observation space by measurements
        self.observations.extend(self.measurement_vars)
        self.lower_obs_bounds.extend([observations[obs][0] for obs in self.measurement_vars])
        self.upper_obs_bounds.extend([observations[obs][1] for obs in self.measurement_vars])
        
        # Check if agent uses regressive states and extend observations with these
        self.is_regressive = False
        if self.regressive_period is not None:
            self.is_regressive = True
            # Do a sanity check
            if self.regressive_period == 0 or self.regressive_period<0:
                raise ValueError(\
                 'The regressive_period cannot be 0 or negative. '\
                 'If you just want to add a measurement variabe to the '\
                 'set of observations it is enough to add it to the '\
                 'observations argument. ')
            self.regressive_vars = self.measurement_vars
        
            # Number of discrete regressive steps. 
            # If regressive_period=3600, and step_period=900
            # then we have 4 regressive steps: 
            # regr_1, regr_2, regr_3, regr_4 (actual not taken here)
            # regr_4 is the time step furthest away in the past
            self.regr_n = int(self.regressive_period/self.step_period)
            
            # Extend observations to have one observation per regressive step
            for obs in self.regressive_vars:
                obs_list = [obs+'_regr_{}'.format(int(i*self.step_period)) for i in range(1,self.regr_n+1)]
                obs_lbou = [observations[obs][0]]*len(obs_list)
                obs_ubou = [observations[obs][1]]*len(obs_list)
                self.observations.extend(obs_list)
                self.lower_obs_bounds.extend(obs_lbou)
                self.upper_obs_bounds.extend(obs_ubou)
        
        # Check if agent uses predictions in state and parse predictive variables
        self.is_predictive = False
        self.predictive_vars = []
        if any([obs in self.all_predictive_vars for obs in observations]):
            self.is_predictive = True
            
            # Do a sanity check
            if self.predictive_period<0:
                raise ValueError(\
                 'The predictive_period cannot be negative. '\
                 'Set the predictive_period to be 0 or higher than 0 ')
            
            # Parse predictive vars
            self.predictive_vars = [obs for obs in observations if \
                                   (obs in self.all_predictive_vars and obs!='time')]
        
            # Number of discrete predictive steps. If predictive_period=0, 
            # then only 1 step is taken: the actual time step. 
            # If predictive_period=3600, and step_period=900
            # then we have 5 predictive steps: 
            # pred_0, pred_1, pred_2, pred_3, pred_4 (actual taken here)
            # pred_4 is the time step furthest away in the future
            self.pred_n = int(self.predictive_period/self.step_period)+1
            
            # Extend observations to have one observation per predictive step
            for obs in self.predictive_vars:
                obs_list = [obs+'_pred_{}'.format(int(i*self.step_period)) for i in range(self.pred_n)]
                obs_lbou = [observations[obs][0]]*len(obs_list)
                obs_ubou = [observations[obs][1]]*len(obs_list)
                self.observations.extend(obs_list)
                self.lower_obs_bounds.extend(obs_lbou)
                self.upper_obs_bounds.extend(obs_ubou)
        
            # If predictive, the margin should be extended        
            self.end_year_margin = self.max_episode_length + self.predictive_period
        
        # Define gym observation space
        self.observation_space = spaces.Box(low  = np.array(self.lower_obs_bounds), 
                                            high = np.array(self.upper_obs_bounds), 
                                            dtype= np.float32)
        
        #--- Populate the name-to-index mapping using the final list ---
        self._obs_name_to_index = {name: i for i, name in enumerate(self.observations)}    
        
        #=============================================================
        # Define action space
        #=============================================================
        # Assert that actions belong to the inputs in the emulator model
        for act in self.actions:
            if not (act in self.all_input_vars.keys()):
                raise ReferenceError(\
                 '"{0}" does not belong to the set of inputs to this '\
                 'emulator model. \n'\
                 'Set of inputs: \n{1}\n'.format(act, list(self.all_input_vars.keys()) ))

        # Parse minimum and maximum values for actions
        self.lower_act_bounds = []
        self.upper_act_bounds = []
        for act in self.actions:
            self.lower_act_bounds.append(self.all_input_vars[act]['Minimum'])
            self.upper_act_bounds.append(self.all_input_vars[act]['Maximum'])
        
        # Define gym action space
        self.action_space = spaces.Box(low  = np.array(self.lower_act_bounds), 
                                       high = np.array(self.upper_act_bounds), 
                                       dtype= np.float32)
        
        if self.render_episodes:
            plt.ion()
            self.fig = plt.gcf()

    def __str__(self):
        '''
        Print a summary of the environment. 
        
        '''
        
        # Get a summary of the environment
        summary = self.get_summary()
        
        # Create a printable string from summary
        s = '\n'
        
        # Iterate over summary, which has two layers of key,value pairs
        for k1,v1 in summary.items():
            s += '='*len(k1) + '\n'
            s += k1 + '\n'
            s += '='*len(k1) + '\n\n'
            for k2,v2 in v1.items():
                s += k2 + '\n'
                s += '-'*len(k2) + '\n'
                s += v2 + '\n\n'

        return s
    
    def get_summary(self):
        '''
        Get a summary of the environment.
        
        Returns
        -------
        summary: OrderedDict
            A dictionary mapping keys and values that fully describe the 
            environment. 
        
        '''
        
        summary = OrderedDict()
        
        summary['BOPTEST CASE INFORMATION'] = OrderedDict()
        summary['BOPTEST CASE INFORMATION']['Test case name'] = pformat(self.name)
        summary['BOPTEST CASE INFORMATION']['All measurement variables'] = pformat(self.all_measurement_vars)
        summary['BOPTEST CASE INFORMATION']['All forecasting variables'] = pformat(list(self.all_predictive_vars.keys()))
        summary['BOPTEST CASE INFORMATION']['All input variables'] = pformat(self.all_input_vars)
        summary['BOPTEST CASE INFORMATION']['Default simulation step (seconds)'] = pformat(self.step_def)
        summary['BOPTEST CASE INFORMATION']['Default scenario'] = pformat(self.scenario_def)
        summary['BOPTEST CASE INFORMATION']['Test case scenario'] = pformat(self.scenario)
        
        summary['GYM ENVIRONMENT INFORMATION'] = OrderedDict()
        summary['GYM ENVIRONMENT INFORMATION']['Observation space'] = pformat(self.observation_space)
        summary['GYM ENVIRONMENT INFORMATION']['Action space'] = pformat(self.action_space)
        summary['GYM ENVIRONMENT INFORMATION']['Is a regressive environment'] = pformat(self.is_regressive)
        summary['GYM ENVIRONMENT INFORMATION']['Is a predictive environment'] = pformat(self.is_predictive)
        summary['GYM ENVIRONMENT INFORMATION']['Regressive period (seconds)'] = pformat(self.regressive_period)
        summary['GYM ENVIRONMENT INFORMATION']['Predictive period (seconds)'] = pformat(self.predictive_period)
        summary['GYM ENVIRONMENT INFORMATION']['Measurement variables used in observation space'] = pformat(self.measurement_vars)
        summary['GYM ENVIRONMENT INFORMATION']['Predictive variables used in observation space'] = pformat(self.predictive_vars)
        summary['GYM ENVIRONMENT INFORMATION']['Sampling time (seconds)'] = pformat(self.step_period)
        summary['GYM ENVIRONMENT INFORMATION']['Random start time'] = pformat(self.random_start_time)
        summary['GYM ENVIRONMENT INFORMATION']['Excluding periods (seconds from the beginning of the year)'] = pformat(self.excluding_periods)
        summary['GYM ENVIRONMENT INFORMATION']['Warmup period for each episode (seconds)'] = pformat(self.warmup_period)
        summary['GYM ENVIRONMENT INFORMATION']['Maximum episode length (seconds)'] = pformat(self.max_episode_length)
        summary['GYM ENVIRONMENT INFORMATION']['Environment reward function (source code)'] = pformat(inspect.getsource(self.get_reward))
        summary['GYM ENVIRONMENT INFORMATION']['Environment hierarchy'] = pformat(inspect.getmro(self.__class__))
        
        return summary

    def save_summary(self, file_name='summary'):
        '''
        Saves the environment summary in a `.json` file. 
        
        Parameters
        ----------
        file_name: string
            File name where the summary will be saved in `.json` format
        
        '''
        
        summary = self.get_summary()
        with open('{}.json'.format(file_name), 'w') as outfile:  
            json.dump(summary, outfile) 
            
    def load_summary(self, file_name='summary'):
        '''
        Loads an environment summary from a `.json` file. 
        
        Parameters
        ----------
        file_name: string
            File in `.json` format from where the summary is to be loaded
        
        Returns
        -------
        summary: OrderedDict
            A summary of an environment
            
        '''
        
        with open(file_name+'.json', 'r') as f:
            summary = json.load(f, object_pairs_hook=OrderedDict)
        
        return summary

    def reset(self, seed=None, options=None):
        '''
        Method to reset the environment. The associated building model is 
        initialized by running the baseline controller for a  
        `self.warmup_period` of time right before `self.start_time`. 
        If `self.random_start_time` is True, a random time is assigned 
        to `self.start_time` such that there are not episodes that overlap
        with the indicated `self.excluding_periods`. This is useful to 
        define testing periods that should not use data from training.   
        
        Parameters
        ----------
        seed: optional int 
            The seed that is used to initialize the environment.
            Currently not being used since the seed is fixed externally. 

        options: optional dict 
            Additional information to specify how the environment is reset.

        Returns
        -------
        observations: numpy array
            Reformatted observations of the initial state which includes measurements and 
            predictions (if any) at the end of the initialization (beginning of the episode).
            It is analogous to the observation returned by the `step` method. 
        info: dictionary
            Additional information for this observation.
            It should be analogous to the info returned by the `step` method. 
         
        '''        

        def find_start_time():
            '''Recursive method to find a random start time out of 
            `excluding_periods`. An episode and an excluding_period that
            are just touching each other are not considered as being 
            overlapped. 
            
            '''
            start_time = random.randint(0+self.bgn_year_margin, 
                                        3.1536e+7-self.end_year_margin)
            episode = (start_time, start_time+self.max_episode_length)
            if self.excluding_periods is not None:
                for period in self.excluding_periods:
                    if episode[0] < period[1] and period[0] < episode[1]:
                        # There is overlapping between episode and this period
                        # Try to find a good starting time again
                        start_time = find_start_time()
            # This point is reached only when a good starting point is found
            return start_time
        
        # Assign random start_time if it is None
        if self.random_start_time:
            self.start_time = find_start_time()
        
        # Initialize the building simulation
        res = requests.put('{0}/initialize/{1}'.format(self.url,self.testid),
                           json={'start_time':int(self.start_time),
                                 'warmup_period':int(self.warmup_period)}).json()['payload']
        
        # Set simulation step
        requests.put('{0}/step/{1}'.format(self.url,self.testid), json={'step':int(self.step_period)})
        
        # Set BOPTEST scenario
        from omegaconf import OmegaConf

        scenario_dict = OmegaConf.to_container(self.scenario, resolve=True)
        requests.put('{0}/scenario'.format(self.url), json=scenario_dict)
        
        # Initialize objective integrand
        self.objective_integrand = 0.
        
        # Get observations at the end of the initialization period
        observations = self.get_observations(res)
        
        # Optionally we can pass additional info, we are not using that for now
        info = {}
        
        self.episode_rewards = []

        return observations, info

    def stop(self):
        '''
        Stop the test case

        '''

        requests.put('{0}/stop/{1}'.format(self.url, self.testid))

    def stop(self):
        '''
        Stop the test case

        '''

        requests.put('{0}/stop/{1}'.format(self.url, self.testid))

    def step(self, action):
        '''
        Advance the simulation one time step
        
        Parameters
        ----------
        action: list
            List of actions computed by the agent to be implemented 
            in this step
            
        Returns
        -------
        observations: numpy array
            Observations at the end of this time step
        reward: float
            Reward for the state-action pair implemented
        terminated: boolean
            Whether a `terminal state` (as defined under the MDP of the task) is reached
        truncated: boolean
            Whether a truncation condition outside the scope of the MDP is satisfied
        info: dictionary
            Additional information for this step
        
        Notes
        -----
        From release 0.25 Gym has performed a major update on its API that solves the ambiguity
        of `done` to distinguish between `terminated` and `truncated`. See:
        https://gymnasium.farama.org/gym_release_notes/index.html#release-0-25-0
        terminated=True if environment terminates (eg. due to task completion, failure etc.)
            In this case further step() calls could return undefined results.
        truncated=True if episode truncates due to a time limit or a reason that is not defined as part of the task MDP.
            Typically a timelimit, but could also be used to indicate agent physically going out of bounds.
            Can be used to end the episode prematurely before a `terminal state` is reached.
        For the application of building energy management we will typically have a truncation 
        since the MDP is normally indefinite by definition. 

        '''
        
        # Initialize inputs to send through BOPTEST Rest API
        u = {}
        
        # Assign values to inputs if any
        for i, act in enumerate(self.actions):
            # Assign value
            u[act] = float(action[i])
            
            # Indicate that the input is active
            u[act.replace('_u','_activate')] = float(1)
                
        # Advance a BOPTEST simulation
        res = requests.post('{0}/advance/{1}'.format(self.url,self.testid), json=u).json()['payload']
        
        # Compute reward of this (state-action-state') tuple
        reward = self.get_reward()
        self.episode_rewards.append(reward)
        
        # Define whether a terminal state (as defined under the MDP of the task) is reached
        terminated = self.compute_terminated(res, reward)
        
        # Optionally we can pass the truncated boolean but not used that for now
        truncated = self.compute_truncated(res, reward)

        # Optionally we can pass additional info, we are not using that for now
        info = {}
        
        # Get observations at the end of this time step
        observations = self.get_observations(res)
        
        # Render episode if finished and requested
        if (terminated or truncated) and self.render_episodes:
            self.render()
        
        return observations, reward, terminated, truncated, info
    
    def render(self, mode='episodes'):
        '''
        Renders the process evolution 
        
        Parameters
        ----------
        mode: string
            Mode to be used for the renderization
        
        '''
        if mode != 'episodes':
            raise NotImplementedError()
        else:
            plt.ion()
            self.fig = plt.gcf()
            self.fig.clear()
            plot_results(self, self.episode_rewards, log_dir=self.log_dir)

    def close(self):
        pass
    
    def get_reward(self):
        '''
        Compute the reward of last state-action-state' tuple. The 
        reward is implemented as the negated increase in the objective
        integrand function. In turn, this objective integrand function 
        is calculated as the sum of the total operational cost plus
        the weighted discomfort. 
        
        Returns
        -------
        Reward: float
            Reward of last state-action-state' tuple
        
        Notes
        -----
        This method is just a default method to compute reward. It can be 
        overridden by defining a child from this class with
        this same method name, i.e. `get_reward`. If a custom reward 
        is defined, it is strongly recommended to derive it using the KPIs
        as returned from the BOPTEST framework, as it is done in this 
        default `get_reward` method. This ensures that all variables 
        that may contribute to any KPI are properly accounted and 
        integrated. 
        
        '''
        
        # Define a relative weight for the discomfort 
        w = 1
        
        # Compute BOPTEST core kpis
        kpis = requests.get('{0}/kpi/{1}'.format(self.url,self.testid)).json()['payload']
        
        # Calculate objective integrand function at this point
        objective_integrand = kpis['cost_tot'] + w*kpis['tdis_tot']
        
        # Compute reward
        reward = -(objective_integrand - self.objective_integrand)
        
        self.objective_integrand = objective_integrand
        
        return reward

    def compute_terminated(self, res, reward=None):
        '''
        Compute whether the episode is terminated as defined by the MDP. 
        `terminated = False` is returned by default as the applications 
        for building energy management are typically indefinite. 
        
        Returns
        -------
        terminated: boolean
            Boolean indicating whether the episode is terminated or not.  
        
        Notes
        -----
        This method can be overridden by defining a child from 
        this class with this same method name, i.e. `compute_terminated`.
        The reward is passed as an argument in case it's necessary to 
        define custom conditions for termination.  
        
        '''
        
        terminated = False

        return terminated

    def compute_truncated(self, res, reward=None):
        '''
        Compute whether the episode is truncated. By default, a 
        maximum episode length is defined and the episode will be finished
        only when the time exceeds this maximum episode length. 
        
        Returns
        -------
        truncated: boolean
            Boolean indicating whether the episode is truncated or not.  
        
        Notes
        -----
        This method is just a default method to determine if an episode is
        truncated or not. It can be overridden by defining a child from 
        this class with this same method name, i.e. `compute_truncated`. Notice
        that the reward for each step is passed here to enable the user to
        access this reward as it may be handy when defining a custom 
        method for `compute_truncated`. 
        
        '''
        
        truncated = res['time'] >= self.start_time + self.max_episode_length
        
        return truncated

    def get_observations(self, res):
        '''
        Get the observations, i.e. the conjunction of measurements,
        regressive and predictive variables if any. Also transforms
        the output to have the right format, with robust error handling.

        Parameters
        ----------
        res: dictionary
            Dictionary mapping simulation variables and their value at the
            end of the last time step (e.g., from /advance or /initialize).

        Returns
        -------
        observations_arr: numpy array
            Reformatted observations that include measurements and
            predictions (if any) at the end of last step.

        Raises
        ------
        RuntimeError
            If critical data (like current measurements or fallbacks)
            is missing, or if fatal errors occur during API calls/processing.
        KeyError
            If essential keys are missing in the input 'res' dictionary
            or API responses where expected.
        '''
        # Use logger if defined, otherwise default logging
        log = getattr(self, 'logger', logging)
        testid = getattr(self, 'testid', 'UNKNOWN_TESTID') # For logging

        # Initialize observations list
        observations = []

        # --- 1. Current Time (Optional) ---
        # Check if 'time' is one of the configured observations (from init)
        is_time_observation = 'time' in self.observations
        if is_time_observation:
            try:
                current_time_val = res.get('time')
                if current_time_val is None:
                     log.error(f"[{testid}] 'time' key missing in 'res' dict needed for observation.")
                     raise KeyError("'time' key missing in BOPTEST response 'res'")
                # Find the index and upper bound for time safely
                time_idx = self.observations.index('time')
                time_upper_bound = self.upper_obs_bounds[time_idx]
                observations.append(current_time_val % time_upper_bound)
            except (ValueError, IndexError) as e: # Handles index() not found or bounds index issue
                 log.error(f"[{testid}] Configuration error finding 'time' bounds/index: {e}")
                 raise RuntimeError(f"Configuration error for 'time' observation: {e}") from e
            except KeyError as e:
                 raise e # Re-raise the KeyError from missing 'time' in res

        # --- 2. Current Measurements ---
        for obs_name in self.measurement_vars:
            try:
                current_meas = res[obs_name] # Direct access, expect it in res
                observations.append(current_meas)
            except KeyError:
                log.error(f"[{testid}] Critical measurement '{obs_name}' missing in 'res': {list(res.keys())}")
                raise KeyError(f"Measurement '{obs_name}' not found in BOPTEST response 'res'")

        # --- 3. Regressive Measurements ---
        if self.is_regressive:
            current_time = res.get('time')
            if current_time is None:
                log.error(f"[{testid}] Missing 'time' in 'res' needed for regression.")
                raise KeyError("'time' key missing in 'res' for regression")

            # Calculate target timestamps for historical data
            regr_target_times = current_time - self.step_period * np.arange(1, self.regr_n + 1)
            start_hist = int(regr_target_times[-1]) # Earliest time needed
            end_hist = int(regr_target_times[0])   # Most recent past time needed

            # Sanity check/adjust time range
            if start_hist < 0:
                log.warning(f"[{testid}] Regressive start time {start_hist} is negative. Clamping to 0.")
                start_hist = 0

            valid_request_range = (start_hist < end_hist)

            for var in self.regressive_vars:
                res_var_payload = None
                res_var_reindexed = None # Final regressive values for this var

                # Get fallback value (current measurement) safely
                fallback_value = res.get(var) # Use .get for safety
                if fallback_value is None:
                    log.warning(f"[{testid}] Fallback value (current) for '{var}' is missing in 'res'. Fallback might fail.")

                # --- API Call ---
                if valid_request_range:
                    try:
                        api_url = f'{self.url}/results/{testid}'
                        api_payload = {'point_names': [var], 'start_time': start_hist, 'final_time': end_hist}
                        log.debug(f"[{testid}] Fetching regressive data: URL={api_url}, Payload={api_payload}")

                        # Use self.session if available, otherwise requests directly
                        requester = getattr(self, 'session', requests)
                        response = requester.put(
                            api_url,
                            json=api_payload,
                            timeout=20
                        )
                        response.raise_for_status()
                        data = response.json()
                        res_var_payload = data.get('payload') # Safely get payload
                        log.debug(f"[{testid}] Received regressive payload for '{var}': Type={type(res_var_payload)}")

                    except requests.exceptions.RequestException as e:
                        log.error(f"[{testid}] API error fetching regressive data for '{var}': {e}")
                    except Exception as e: # Catch JSON errors etc.
                        log.error(f"[{testid}] Error during regressive fetch/parse for '{var}': {e}", exc_info=True)
                    # On error, res_var_payload remains None, triggers fallback below
                else:
                     log.warning(f"[{testid}] Invalid time range ({start_hist} to {end_hist}). Skipping API call for '{var}'.")

                # --- Process Payload & Interpolate (or Fallback) ---
                data_valid_for_interp = False
                single_point_value = None

                if res_var_payload is not None and isinstance(res_var_payload, dict) and \
                   'time' in res_var_payload and var in res_var_payload and \
                   isinstance(res_var_payload.get('time'), list) and isinstance(res_var_payload.get(var), list):
                   num_points = len(res_var_payload['time'])
                   if num_points >= 2:
                       data_valid_for_interp = True
                   elif num_points == 1:
                       single_point_value = res_var_payload[var][0]
                       log.warning(f"[{testid}] Only 1 point received for regressive '{var}'. Using constant: {single_point_value}")
                       res_var_reindexed = np.full(self.regr_n, fill_value=single_point_value, dtype=np.float32)
                   # If num_points == 0, falls through to fallback

                if data_valid_for_interp:
                    try:
                        hist_times = res_var_payload['time']
                        hist_values = res_var_payload[var]
                        log.debug(f"[{testid}] Interpolating for '{var}' using {len(hist_times)} points.")
                        f_interp = interpolate.interp1d(hist_times, hist_values,
                                                        kind='linear',
                                                        bounds_error=False,
                                                        fill_value="extrapolate")
                        res_var_reindexed = f_interp(regr_target_times)

                        # Check for NaNs post-interpolation
                        if np.isnan(res_var_reindexed).any():
                            log.warning(f"[{testid}] NaNs found after interpolating '{var}'. Trying fallback fill.")
                            if fallback_value is not None and np.isfinite(fallback_value):
                                res_var_reindexed[np.isnan(res_var_reindexed)] = fallback_value
                                if np.isnan(res_var_reindexed).any(): # Should not happen if fallback is finite
                                     log.error(f"[{testid}] Failed to fill NaNs for '{var}' even with fallback {fallback_value}.")
                                     raise RuntimeError(f"Unfillable NaNs after interpolation for {var}")
                            else:
                                log.error(f"[{testid}] Cannot fill NaNs for '{var}': Fallback value missing or non-finite.")
                                raise RuntimeError(f"NaNs in interpolation and no valid fallback for {var}")

                    except (ValueError, IndexError, Exception) as interp_e: # Catch broad interp errors
                        log.error(f"[{testid}] Interpolation failed for '{var}': {interp_e}. Using fallback.", exc_info=True)
                        res_var_reindexed = None # Signal fallback needed

                # --- Apply Fallback if Needed ---
                # Fallback triggered if: API failed (payload is None), 0 points, interp failed (reindexed is None)
                if res_var_reindexed is None:
                    log_reason = "API/payload error" if res_var_payload is None else "Interpolation error/0 points"
                    log.warning(f"[{testid}] {log_reason} for regressive '{var}'. Using fallback (current value).")
                    if fallback_value is not None:
                        res_var_reindexed = np.full(self.regr_n, fill_value=fallback_value, dtype=np.float32)
                    else:
                        log.error(f"[{testid}] Cannot apply fallback for '{var}': Current value missing in 'res'.")
                        raise RuntimeError(f"Missing regressive data and current value fallback for {var}")

                # --- Append Regressive Values ---
                observations.extend(list(res_var_reindexed))

        # --- 4. Predictive Forecasts ---
        if self.is_predictive:
            predictions_payload = None
            try:
                api_url = f'{self.url}/forecast/{testid}'
                api_payload = {'point_names': self.predictive_vars,
                               'horizon': int(self.predictive_period),
                               'interval': int(self.step_period)}
                log.debug(f"[{testid}] Fetching forecast data: URL={api_url}, Payload={api_payload}")

                requester = getattr(self, 'session', requests)
                response = requester.put(
                    api_url,
                    json=api_payload,
                    timeout=20
                )
                response.raise_for_status()
                data = response.json()
                predictions_payload = data.get('payload')
                log.debug(f"[{testid}] Received forecast payload: Type={type(predictions_payload)}")

            except requests.exceptions.RequestException as e:
                log.error(f"[{testid}] API error fetching forecast data: {e}")
                raise RuntimeError(f"Failed to get forecast data: {e}") from e
            except Exception as e:
                log.error(f"[{testid}] Error fetching/parsing forecast data: {e}", exc_info=True)
                raise RuntimeError(f"Error processing forecast data: {e}") from e

            # Validate and extract predictive data
            if predictions_payload is None or not isinstance(predictions_payload, dict):
                 log.error(f"[{testid}] Invalid or missing 'payload' in forecast response.")
                 raise RuntimeError("Received invalid forecast payload from API.")

            for var in self.predictive_vars:
                if var not in predictions_payload:
                    log.error(f"[{testid}] Predictive var '{var}' missing in forecast payload: {list(predictions_payload.keys())}")
                    raise RuntimeError(f"Forecast data missing for requested variable: {var}")

                forecast_values = predictions_payload[var]
                expected_len = self.pred_n # self.pred_n = int(self.predictive_period/self.step_period)+1

                if not isinstance(forecast_values, list) or len(forecast_values) != expected_len:
                    log.error(f"[{testid}] Forecast for '{var}' wrong type/length. "
                              f"Expected list[{expected_len}], got {type(forecast_values)} "
                              f"len={len(forecast_values) if isinstance(forecast_values, list) else 'N/A'}")
                    raise RuntimeError(f"Invalid forecast data structure/length for: {var}")

                # Append the forecast steps
                observations.extend(forecast_values)


        # --- 5. Final Formatting & Validation ---
        try:
            # Convert collected observations to a numpy array
            observations_arr = np.array(observations, dtype=np.float32)
        except ValueError as e:
             # This might happen if fallback values or API responses were non-numeric strings, etc.
             log.error(f"[{testid}] Failed to convert final observations list to float32 array: {e}", exc_info=True)
             log.error(f"[{testid}] Content that failed conversion: {observations}")
             raise RuntimeError(f"Could not create final observation array: {e}")

        # Store the raw array before any potential wrapping/normalization
        self.last_raw_observation = observations_arr.copy() # Use copy() for safety

        # Final check for NaN/Inf
        if not np.all(np.isfinite(observations_arr)):
            nan_inf_indices = np.where(~np.isfinite(observations_arr))[0]
            log.error(f"[{testid}] Non-finite values (NaN/Inf) detected in final observation array at indices {nan_inf_indices}!")
            log.error(f"[{testid}] Array content: {observations_arr}")
            # Depending on the RL lib, NaNs/Infs can crash training. Raising is safest.
            raise ValueError(f"Non-finite values detected in final observation vector for {testid}")

        log.debug(f"[{testid}] Final observations generated. Shape: {observations_arr.shape}")
        return observations_arr
    
    def get_kpis(self):
        '''Auxiliary method to get the so-colled core KPIs as computed in 
        the BOPTEST framework. This is handy when evaluating performance 
        of an agent in this environment. 
        
        '''
        
        # Compute BOPTEST core kpis
        kpis = requests.get('{0}/kpi/{1}'.format(self.url, self.testid)).json()['payload']
        
        return kpis
    
    def reformat_expert_traj(self, file_path='data.csv'):
        '''
        Reformats expert trajectory from a csv file to the npz format 
        required by Stable Baselines algorithms to be pre-trained.   
        
        Parameters
        ----------
        file_path: string
            path to csv file containing data
            
        Returns
        -------
        numpy_dict: numpy dictionary
            Numpy dictionary with the reformatted data
        
        Notes
        -----
        The resulting reformatted data considers only one episode from
        a long trajectory (a long time series). No recurrent policies 
        supported (mask and state not defined). 
        
        '''
        
        # We consider only one episode of index 0 that is never done
        n_episodes = 1
        ep_idx = 0
        done = False
        
        # Initialize data in the episode
        actions = []
        observations = []
        rewards = []
        episode_returns = np.zeros((n_episodes,))
        episode_starts = []
        
        # Initialize the only episode that we use
        episode_starts.append(True)
        reward_sum = 0.0

        df = pd.read_csv(file_path)
        for row in df.index:
            # Retrieve step information from csv
            obs     = df.loc[row, self.observations]
            action  = df.loc[row, self.actions]
            reward  = df.loc[row, self.reward]
            
            if obs.hasnans or action.hasnans or reward.hasnans:
                raise ValueError('Nans found in row {}'.format(row))
            
            # Append to data
            observations.append(np.array(obs))
            actions.append(np.array(action))
            rewards.append(np.array(reward))
            episode_starts.append(np.array(done))
            
            reward_sum += reward
        
        # This is hard coded as we only support one episode so far but
        # here we could implement some functionality for creating different 
        # episodes from csv data
        done = True
        if done:
            episode_returns[ep_idx] = reward_sum
            reward_sum = 0.0
    
        if isinstance(self.observation_space, spaces.Box):
            observations = np.concatenate(observations).reshape((-1,) + self.observation_space.shape)
        elif isinstance(self.observation_space, spaces.Discrete):
            observations = np.array(observations).reshape((-1, 1))
    
        if isinstance(self.action_space, spaces.Box):
            actions = np.concatenate(actions).reshape((-1,) + self.action_space.shape)
        elif isinstance(self.action_space, spaces.Discrete):
            actions = np.array(actions).reshape((-1, 1))
    
        rewards = np.array(rewards)
        episode_starts = np.array(episode_starts[:-1])
    
        assert len(observations) == len(actions)
    
        numpy_dict = {
            'actions': actions,
            'obs': observations,
            'rewards': rewards,
            'episode_returns': episode_returns,
            'episode_starts': episode_starts
        } 
    
        for key, val in numpy_dict.items():
            print(key, val.shape)
    
        np.savez(file_path.split('.')[-2], **numpy_dict)
        
        return numpy_dict 

class DiscretizedObservationWrapper(gym.ObservationWrapper):
    '''This wrapper converts the Box observation space into a Discrete 
    observation space.
    
    Notes
    -----
    The concept of wrappers is very powerful, with which we are capable 
    to customize observation, action, step function, etc. of an env. 
    No matter how many wrappers are applied, `env.unwrapped` always gives 
    back the internal original environment object. Typical use:
    `env = BoptestGymEnv()`
    `env = DiscretizedObservationWrapper(env, n_bins_obs=10)`
    
    '''
    
    def __init__(self, env, n_bins_obs=10, outs_are_bins=True):
        '''
        Constructor
        
        Parameters
        ----------
        env: gym.Env
            Original gym environment
        n_bins_obs: integer
            Number of bins to be used in the transformed observation 
            space for each observation.
        outs_are_bins: boolean
            Decides whether to include the outer observation spaces
            as bins or not. For example, in the case that we observe
            temperature and it has bounds [21,24] degC and
            `n_bins_obs = 3`
            If outs_are_bins=False, then the bins are defined by
            `val_bins_obs=[21,22,23,24]`, and an error is raised when the
            observation goes out of bounds.
            If outs_are_bins=True, then the bins are defined by
            `val_bins_obs=[-inf,21,24,+inf]`.
            Notice in both cases the observation space dimension
            equals n_bins_obs=3.
        '''
        
        # Construct from parent class
        super().__init__(env)
        
        # Assign attributes (env already assigned)
        self.n_bins_obs     = n_bins_obs
        self.outs_are_bins  = outs_are_bins

        # Assert that original observation space is a Box space
        assert isinstance(env.observation_space, spaces.Box), 'This wrapper only works with continuous action space (spaces.Box)'
        
        # Get observation space bounds
        self.low  = self.observation_space.low
        self.high = self.observation_space.high
        
        # Calculate dimension of observation space
        n_obs = self.low.flatten().shape[0]

        # Set number of bins within bounds
        n_within_bounds = n_bins_obs - 1 if self.outs_are_bins else n_bins_obs + 1

        # Obtain values of discretized observation space
        self.val_bins_obs = [np.linspace(l, h, n_within_bounds) for l, h in
                             zip(self.low.flatten(), self.high.flatten())]
        
        # Instantiate discretized observation space
        self.observation_space = spaces.Discrete(n_bins_obs ** n_obs)

    def observation(self, observation):
        '''
        This method accepts a single parameter (the 
        observation to be modified) and returns the modified observation.
        
        Parameters
        ----------
        observation: 
            Observation in the original environment observation space format 
            to be modified.
        
        Returns
        -------
            Modified observation returned by the wrapped environment. 
        
        Notes
        -----
        To better understand what this method needs to do, see how the 
        `gym.ObservationWrapper` parent class is doing in `gym.core`:
        
        '''
        
        # Get the bin indexes for each element of this observation
        indexes = [np.digitize([x], bins)[0]
                  for x, bins in zip(observation.flatten(), self.val_bins_obs)]

        # Check if out of bounds when self.outs_are_bins=False
        if not self.outs_are_bins:
            for i, obs in enumerate(observation):
                if obs < self.low[i] or obs > self.high[i]:
                    raise ValueError( \
                        'Observation {0} is out of bounds [{1},{2}] ' \
                        'The discretized environment has been configured ' \
                        'not to include the space out of these bounds as ' \
                        'bins. You may want to set outs_are_bins=True ' \
                        'in the DiscretizedObservationWrapper.'.format(obs, self.low, self.high))
            else:
                indexes = [i - 1 for i in indexes]

        # Convert to one number for the wrapped environment
        observation_wrapper = sum([index * ((self.n_bins_obs + 1) ** obs_i) for obs_i, index in enumerate(indexes)])
        
        return observation_wrapper
    
class DiscretizedActionWrapper(gym.ActionWrapper):
    '''
    Converts Box action space to Discrete, implements safety action masking.

    Safety Rule: Prevent heat pump activation (any level > 0) if the zone
    temperature is within `safety_margin` of the active upper OR lower comfort
    setpoint. The active comfort setpoints are determined by inferring
    occupancy status from the measured heating setpoint (`reaTSetHea_y`).

    Compatible with Stablebaselines3 contrib MaskablePPO.
    '''
    def __init__(self,
                 env,
                 n_bins_act=10,
                 masking_enabled=True,
                 # Arguments below are now primarily for passing constants defined above
                 zone_temp_obs_name=ZONE_TEMP_OBS_NAME,
                 occupancy_proxy_obs_name=OCCUPANCY_PROXY_OBS_NAME, # Use proxy name
                 heating_action_name=HEATING_ACTION_NAME,
                 occupied_upper_sp=OCCUPIED_UPPER_SP_K,
                 occupied_lower_sp=OCCUPIED_LOWER_SP_K,
                 unoccupied_upper_sp=UNOCCUPIED_UPPER_SP_K,
                 unoccupied_lower_sp=UNOCCUPIED_LOWER_SP_K,
                 safety_margin=SAFETY_MARGIN_K,
                 assume_obs_in_kelvin=ASSUME_OBS_IN_KELVIN):
        '''Constructor'''
        super().__init__(env)

        # --- Basic Setup ---
        assert isinstance(env.action_space, spaces.Box), \
            'This wrapper only works with continuous action space (spaces.Box)'
        self.n_bins_act = n_bins_act
        self.num_choices_per_dim = n_bins_act + 1
        low, high = self.env.action_space.low, self.env.action_space.high
        self.n_act = low.flatten().shape[0]
        self.val_bins_act = [np.linspace(l, h, self.num_choices_per_dim) for l, h in
                             zip(low.flatten(), high.flatten())]
        self.action_space = spaces.Discrete(self.num_choices_per_dim ** self.n_act)
        print(f"DiscretizedActionWrapper: Discrete space size {self.action_space.n}.")

        # --- Masking Configuration ---
        self.masking_enabled = masking_enabled
        self._masking_possible = False # Internal flag if setup is complete
        self.zone_temp_obs_index = -1
        self.occupancy_proxy_obs_index = -1 # Index for the proxy observation
        self.heating_action_cont_index = -1
        # Store names from arguments
        self.heating_action_name = heating_action_name
        self.zone_temp_obs_name = zone_temp_obs_name
        self.occupancy_proxy_obs_name = occupancy_proxy_obs_name

        # Store comfort setpoints and margin for the MASKING RULE
        self.occupied_upper_sp = occupied_upper_sp
        self.occupied_lower_sp = occupied_lower_sp
        self.unoccupied_upper_sp = unoccupied_upper_sp
        self.unoccupied_lower_sp = unoccupied_lower_sp
        self.safety_margin = safety_margin
        self.assume_obs_in_kelvin = assume_obs_in_kelvin

        # --- Store Setpoints for OCCUPANCY DETERMINATION (using constants) ---
        self.occupied_heat_sp_val = OCCUPIED_HEATING_SETPOINT_K
        self.unoccupied_heat_sp_val = UNOCCUPIED_HEATING_SETPOINT_K
        self.setpoint_tolerance = SETPOINT_COMPARISON_TOLERANCE
        # --------------------------------------------------------------------

        unit = "Kelvin" if assume_obs_in_kelvin else "Celsius"
        print(f"Info [Masking]: Safety margin {self.safety_margin:.2f} {unit}.")
        print(f"Info [Masking]: Occupied Comfort SPs [{self.occupied_lower_sp:.1f}, {self.occupied_upper_sp:.1f}] {unit}.")
        print(f"Info [Masking]: Unoccupied Comfort SPs [{self.unoccupied_lower_sp:.1f}, {self.unoccupied_upper_sp:.1f}] {unit}.")
        print(f"Info [Masking]: Using '{self.occupancy_proxy_obs_name}' as occupancy proxy.")
        print(f"Info [Masking]: Expecting Occupied Heat SP ~{self.occupied_heat_sp_val:.1f} K, Unoccupied ~{self.unoccupied_heat_sp_val:.1f} K")

        print(f"DEBUG [Masking Init]: Starting masking setup checks...")

        if self.masking_enabled:
            print(f"DEBUG [Masking Init]: Starting simplified setup checks...")
            base_env = None
            try:
                # print(f"DEBUG [Masking Init]: Accessing self.env.unwrapped (self.env is {type(self.env)})...")
                base_env = self.env.unwrapped
                # print(f"DEBUG [Masking Init]: Got base_env of type {type(base_env)}")

                # 1. Check for Observation Map (_obs_name_to_index)
                # print(f"DEBUG [Masking Init]: Checking for _obs_name_to_index...")
                if hasattr(base_env, '_obs_name_to_index') and isinstance(base_env._obs_name_to_index, dict):
                    obs_map = base_env._obs_name_to_index
                    # print(f"DEBUG [Masking Init]: Found _obs_name_to_index with {len(obs_map)} keys.")
                    # Find zone temp index
                    if self.zone_temp_obs_name in obs_map:
                        self.zone_temp_obs_index = obs_map[self.zone_temp_obs_name]
                        # print(f"DEBUG [Masking Init]: Found zone temp index: {self.zone_temp_obs_index}")
                    else:
                        warnings.warn(f"[DiscretizedActionWrapper] Observation '{self.zone_temp_obs_name}' not found. Masking disabled.", RuntimeWarning)
                        self.masking_enabled = False
                    # Find occupancy proxy index
                    if self.masking_enabled and self.occupancy_proxy_obs_name in obs_map:
                        self.occupancy_proxy_obs_index = obs_map[self.occupancy_proxy_obs_name]
                        # print(f"DEBUG [Masking Init]: Found occupancy proxy index: {self.occupancy_proxy_obs_index}")
                    elif self.masking_enabled:
                         warnings.warn(f"[DiscretizedActionWrapper] Occupancy Proxy Obs '{self.occupancy_proxy_obs_name}' not found. Masking disabled.", RuntimeWarning)
                         self.masking_enabled = False
                else:
                    warnings.warn(f"[DiscretizedActionWrapper] Base env missing or invalid '_obs_name_to_index'. Masking disabled.", RuntimeWarning)
                    self.masking_enabled = False

                # 2. Check for last_raw_observation attribute (only warn)
                if self.masking_enabled and not hasattr(base_env, 'last_raw_observation'):
                     warnings.warn("[DiscretizedActionWrapper] Base env missing 'last_raw_observation' attribute. Masking might fail on first step.", RuntimeWarning)

                # 3. Check if obs indices were found
                if self.masking_enabled and (self.zone_temp_obs_index == -1 or self.occupancy_proxy_obs_index == -1):
                     warnings.warn("[DiscretizedActionWrapper] Masking setup failed in __init__ (missing obs indices). Masking disabled.", RuntimeWarning)
                     self.masking_enabled = False # Ensure disabled

            except AttributeError as e:
                 warnings.warn(f"[DiscretizedActionWrapper] AttributeError accessing unwrapped environment in __init__. Masking disabled. Error: {e}", RuntimeWarning)
                 self.masking_enabled = False
            except Exception as e:
                 warnings.warn(f"[DiscretizedActionWrapper] Unexpected error during masking __init__. Masking disabled. Error: {e}", RuntimeWarning)
                 self.masking_enabled = False

        # Final status message
        if not self.masking_enabled:
            print("Info [Masking Init]: Masking is disabled or initial setup failed.")
        else:
            # Indicate that the setup is deferred, success depends on action_masks
            print("Info [Masking Init]: Masking is enabled, setup checks deferred to action_masks.")


    def _get_indices(self, action_wrapper):
        """ Converts flat discrete action index to multi-dimensional bin indices. """
        if not (0 <= action_wrapper < self.action_space.n):
             raise ValueError(f"Action ({action_wrapper}) out of bounds [0, {self.action_space.n-1}]")
        # Correct way to unravel for this encoding scheme (reverse of encoding in original code)
        indices = []
        q = action_wrapper
        base = self.num_choices_per_dim
        for _ in range(self.n_act):
             indices.append(q % base)
             q //= base
        return indices[::-1] # Return in the correct order


    def action(self, action_wrapper):
        ''' Converts discrete action index to continuous action vector. '''
        indices = self._get_indices(action_wrapper)
        action_cont = [self.val_bins_act[i][index] for i, index in enumerate(indices)]
        # Clip to ensure bounds are strictly respected (due to potential float precision issues)
        action_cont_clipped = np.clip(action_cont, self.env.action_space.low, self.env.action_space.high)
        return np.asarray(action_cont_clipped).astype(self.env.action_space.dtype)

    # --- MASKING METHOD ---
    def action_masks(self) -> np.ndarray:
        """
        Calculates the action mask based on the safety rule using setpoint proxy.
        Finds heating action index and calculates heating actions on the fly if needed.
        Returns a boolean numpy array (True=valid, False=invalid).
        """
        mask = np.ones(self.action_space.n, dtype=bool)

        # --- Initial Check ---
        if not self.masking_enabled:
            return mask.astype(bool)

        # --- Defer Heating Index Lookup ---
        # We only need to do this once per environment instance
        if self.heating_action_cont_index == -1:
            # print("DEBUG [action_masks]: First call or heating index not found yet. Trying to find...")
            try:
                # Access unwrapped env reliably HERE
                base_env = self.env.unwrapped
                if hasattr(base_env, 'actions') and isinstance(base_env.actions, list):
                     self.heating_action_cont_index = base_env.actions.index(self.heating_action_name)
                     # print(f"DEBUG [action_masks]: Found heating action index: {self.heating_action_cont_index}")
                     # Now that we have the action index, check if we also have obs indices
                     if self.zone_temp_obs_index != -1 and self.occupancy_proxy_obs_index != -1:
                         self._masking_possible = True # Enable masking logic
                         print(f"Info [action_masks]: Heating index found ({self.heating_action_cont_index}). Masking is now possible.")
                     else:
                          warnings.warn(f"[action_masks] Found heating index, but critical observation indices missing. Masking cannot be activated.", RuntimeWarning)
                          self.masking_enabled = False # Disable permanently
                else:
                     warnings.warn(f"[action_masks] Base env missing/invalid 'actions' list when needed. Masking disabled.", RuntimeWarning)
                     self.masking_enabled = False
            except (ValueError, AttributeError, Exception) as e:
                warnings.warn(f"[action_masks] Error finding heating action index '{self.heating_action_name}'. Masking disabled. Error: {e}", RuntimeWarning)
                self.masking_enabled = False

            # If lookup failed or we are still disabled, return default mask
            if not self.masking_enabled or not self._masking_possible:
                 return mask.astype(bool)

        # --- Check again if masking is possible (might have failed above) ---
        if not self._masking_possible:
             return mask.astype(bool)

        # --- Get Observation ---
        # Access unwrapped env again for the observation
        try:
            last_obs = self.env.unwrapped.last_raw_observation
            if last_obs is None:
                # This can happen right after reset before first step in some setups
                # warnings.warn("action_masks called when last_raw_observation is None. Returning default mask.", RuntimeWarning)
                return mask.astype(bool)
        except AttributeError:
             warnings.warn("[action_masks] Failed to get last_raw_observation from unwrapped env. Returning default mask.", RuntimeWarning)
             return mask.astype(bool)


        # --- Calculate Mask Logic ---
        try:
            # Check indices are valid before using them
            if self.zone_temp_obs_index < 0 or self.occupancy_proxy_obs_index < 0 :
                 warnings.warn(f"[action_masks] Invalid observation indices ({self.zone_temp_obs_index}, {self.occupancy_proxy_obs_index}). Skipping mask calculation.", RuntimeWarning)
                 return mask.astype(bool) # Return default mask

            zone_temp = last_obs[self.zone_temp_obs_index]
            current_heating_sp_obs = last_obs[self.occupancy_proxy_obs_index]

            # Determine occupancy (same as before)
            is_occupied = False
            if abs(current_heating_sp_obs - self.occupied_heat_sp_val) <= self.setpoint_tolerance:
                is_occupied = True

            # Determine active comfort setpoints (same as before)
            if is_occupied:
                current_upper_sp = self.occupied_upper_sp
                current_lower_sp = self.occupied_lower_sp
            else:
                current_upper_sp = self.unoccupied_upper_sp
                current_lower_sp = self.unoccupied_lower_sp

            # Check safety conditions (same as before)
            temp_near_upper = (zone_temp >= current_upper_sp - self.safety_margin)
            temp_near_lower = (zone_temp <= current_lower_sp + self.safety_margin)

            # If temperature is too close, apply mask
            if temp_near_upper or temp_near_lower:
                 # Check heating action index is valid before using
                 if self.heating_action_cont_index < 0:
                      warnings.warn(f"[action_masks] Invalid heating action index ({self.heating_action_cont_index}). Cannot apply mask.", RuntimeWarning)
                      return mask.astype(bool) # Return default mask

                 # --- Calculate heating actions ON THE FLY ---
                 heating_on_action_indices_now = []
                 heating_off_bin_index = 0 # Assuming 0 is off
                 for discrete_action_idx in range(self.action_space.n):
                     bin_indices = self._get_indices(discrete_action_idx)
                     # Additional safety check for index validity
                     if self.heating_action_cont_index >= len(bin_indices):
                          warnings.warn(f"Invalid heating_action_cont_index {self.heating_action_cont_index} vs bin_indices length {len(bin_indices)} for action {discrete_action_idx}. Skipping.", RuntimeWarning)
                          continue

                     heating_bin_index = bin_indices[self.heating_action_cont_index]
                     if heating_bin_index != heating_off_bin_index:
                         heating_on_action_indices_now.append(discrete_action_idx)
                 # -----------------------------------------

                 if heating_on_action_indices_now:
                     mask[heating_on_action_indices_now] = False
                     print(f"DEBUG Masking Applied: T={zone_temp:.1f}. Masking {len(heating_on_action_indices_now)} actions.")


        except IndexError:
             # This might happen if observation shape changes unexpectedly
             warnings.warn(f"[action_masks] IndexError accessing observation (T_idx={self.zone_temp_obs_index}, OccProxy_idx={self.occupancy_proxy_obs_index}). Mask calculation skipped.", RuntimeWarning)
             return np.ones(self.action_space.n, dtype=bool)
        except Exception as e:
            warnings.warn(f"[action_masks] Error during mask calculation logic: {e}. Mask calculation skipped.", RuntimeWarning)
            return np.ones(self.action_space.n, dtype=bool)

        # Ensure at least one action is valid (same as before)
        if not np.any(mask):
            warnings.warn("[action_masks] All actions were masked! Re-enabling action 0.", RuntimeWarning)
            mask[0] = True

        return mask.astype(bool)


    # --- OVERRIDE step and reset to include the mask ---
    def step(self, action_wrapper):
        """ Steps the environment using the discrete action, calculates and adds the action mask to the info dict. """
        continuous_action = self.action(action_wrapper)
        # Step the base environment (this updates self.env.unwrapped.last_raw_observation)
        observation, reward, terminated, truncated, info = self.env.step(continuous_action)
        # Calculate mask based on the NEW observation
        current_mask = self.action_masks()
        # Add the mask to the info dictionary
        if info is None: info = {} # Should not happen with Monitor, but safety check
        info["action_mask"] = current_mask
        return observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """ Resets the environment and adds the initial action mask to the info dict. """
        # Reset the base environment (this updates self.env.unwrapped.last_raw_observation)
        observation, info = self.env.reset(seed=seed, options=options)
        # Calculate mask based on the INITIAL observation
        initial_mask = self.action_masks()
        # Add the mask to the info dictionary
        if info is None: info = {} # Might be None depending on base env reset implementation
        info["action_mask"] = initial_mask
        return observation, info
      
class NormalizedObservationWrapper(gym.ObservationWrapper):
    '''This wrapper normalizes the values of the observation space to lie
    between -1 and 1. Normalization can significantly help with convergence
    speed. 
    
    Notes
    -----
    The concept of wrappers is very powerful, with which we are capable 
    to customize observation, action, step function, etc. of an env. 
    No matter how many wrappers are applied, `env.unwrapped` always gives 
    back the internal original environment object. Typical use:
    `env = BoptestGymEnv()`
    `env = NormalizedObservationWrapper(env)`
    
    '''
    
    def __init__(self, env):
        '''
        Constructor
        
        Parameters
        ----------
        env: gym.Env
            Original gym environment
        
        '''
        
        # Construct from parent class
        super().__init__(env)
        
    def observation(self, observation):
        '''
        This method accepts a single parameter (the 
        observation to be modified) and returns the modified observation.
        
        Parameters
        ----------
        observation: 
            Observation in the original environment observation space format 
            to be modified. 
        
        Returns
        -------
            Modified observation returned by the wrapped environment. 
        
        Notes
        -----
        To better understand what this method needs to do, see how the 
        `gym.ObservationWrapper` parent class is doing in `gym.core`:
        
        '''
        
        # Convert to one number for the wrapped environment
        observation_wrapper = 2*(observation - self.observation_space.low)/\
            (self.observation_space.high-self.observation_space.low)-1
        
        return observation_wrapper
     
class NormalizedActionWrapper(gym.ActionWrapper):
    '''This wrapper normalizes the values of the action space to lie
    between -1 and 1. Normalization can significantly help with convergence
    speed. 
    
    Notes
    -----
    The concept of wrappers is very powerful, with which we are capable 
    to customize observation, action, step function, etc. of an env. 
    No matter how many wrappers are applied, `env.unwrapped` always gives 
    back the internal original environment object. Typical use:
    `env = BoptestGymEnv()`
    `env = NormalizedActionWrapper(env)`
    
    '''
    
    def __init__(self, env):
        '''
        Constructor
        
        Parameters
        ----------
        env: gym.Env
            Original gym environment
        
        '''
        
        # Construct from parent class
        super().__init__(env)
        
        # Assert that original observation space is a Box space
        assert isinstance(self.unwrapped.action_space, spaces.Box), 'This wrapper only works with continuous action space (spaces.Box)'
        
        # Store low and high bounds of action space
        self.low    = self.unwrapped.action_space.low
        self.high   = self.unwrapped.action_space.high
        
        # Redefine action space to lie between [-1,1]
        self.action_space = spaces.Box(low = -1, 
                                       high = 1,
                                       shape=self.unwrapped.action_space.shape, 
                                       dtype= np.float32)        
        
    def action(self, action_wrapper):
        '''This method accepts a single parameter (the modified action
        in the wrapper format) and returns the action to be passed to the 
        original environment. Thus, this method basically rescales the  
        action inside the environment.
        
        Parameters
        ----------
        action_wrapper: 
            Action in the modified environment action space format 
            to be reformulated back to the original environment format.
        
        Returns
        -------
            Action in the original environment format.  
        
        Notes
        -----
        To better understand what this method needs to do, see how the 
        `gym.ActionWrapper` parent class is doing in `gym.core`:
        
        '''
        
        return self.low + (0.5*(action_wrapper+1.0)*(self.high-self.low))

class BoptestGymEnvRewardClipping(BoptestGymEnv):
    '''Boptest gym environment that redefines the reward function to be a 
    clipped reward function penalizing cost and discomfort. 
    
    '''
    
    def get_reward(self):
        '''Clipped reward function that has the value either -1 when
        there is any cost/discomfort, or 0 where there is not cost 
        nor discomfort. This would be the simplest reward to learn for
        an agent. 
        
        Returns
        -------
        reward: float
            Reward of last state-action-state' tuple
        
        '''
        
        # Compute BOPTEST core kpis
        kpis = requests.get('{0}/kpi/{1}'.format(self.url, self.testid)).json()['payload']
        
        # Calculate objective integrand function at this point
        objective_integrand = kpis['cost_tot'] + kpis['tdis_tot']
        
        # Compute reward
        reward = -(objective_integrand - self.objective_integrand)
        
        # Filter to be either -1 or 0
        reward = np.sign(reward)
        
        self.objective_integrand = objective_integrand
        
        return reward

class BoptestGymEnvRewardWeightCost(BoptestGymEnv):
    '''Boptest gym environment that redefines the reward function to 
    weight more the operational cost when compared with the default reward
    function. 
    
    '''
    
    def get_reward(self):
        '''Custom reward function that penalizes less the discomfort
        and thus more the operational cost.
        
        Returns
        -------
        reward: float
            Reward of last state-action-state' tuple
        
        '''
        
        # Define relative weight for discomfort 
        w = 0.1
        
        # Compute BOPTEST core kpis
        kpis = requests.get('{0}/kpi/{1}'.format(self.url, self.testid)).json()['payload']
        
        # Calculate objective integrand function at this point
        objective_integrand = kpis['cost_tot'] + w*kpis['tdis_tot']
        
        # Compute reward
        reward = -(objective_integrand - self.objective_integrand)
        
        self.objective_integrand = objective_integrand
        
        return reward
    
class BoptestGymEnvRewardWeightDiscomfort(BoptestGymEnv):
    '''Boptest gym environment that redefines the reward function to 
    weight more the discomfort when compared with the default reward
    function. 
    
    '''
    
    def get_reward(self):
        '''Custom reward function that penalizes more the discomfort
        and thus more the operational cost.
        
        Returns
        -------
        reward: float
            Reward of last state-action-state' tuple
        
        '''
        
        # Define relative weight for discomfort 
        w = 10
        
        # Compute BOPTEST core kpis
        kpis = requests.get('{0}/kpi/{1}'.format(self.url, self.testid)).json()['payload']
        
        # Calculate objective integrand function at this point
        objective_integrand = kpis['cost_tot'] + w*kpis['tdis_tot']
        
        # Compute reward
        reward = -(objective_integrand - self.objective_integrand)
        
        self.objective_integrand = objective_integrand
        
        return reward
    
class BoptestGymEnvVariableEpisodeLength(BoptestGymEnv):
    '''Boptest gym environment that redefines the reward function to 
    weight more the operational cost when compared with the default reward
    function. 
    
    '''
    
    def compute_truncated(self, res, reward=None, 
                          objective_integrand_threshold=0.1):
        '''Custom method to determine that the episode is truncated not only 
        when the maximum episode length is exceeded but also when the 
        objective integrand overpasses a certain threshold. The latter is
        useful to early stop agent strategies that do not work, hence
        avoiding unnecessary steps and leading to improved sampling 
        efficiency. 
        
        Returns
        -------
        truncated: boolean
            Boolean indicating whether the episode is done or not.  
        
        '''
        
        truncated =  (res['time'] >= self.start_time + self.max_episode_length)\
                     or \
                     (self.objective_integrand >= objective_integrand_threshold)
        
        return truncated

class SaveAndTestCallback(BaseCallback):
    '''
    Callback for saving a model (the check is done every `check_freq` 
    steps) based on the training reward (in practice, we recommend using 
    `EvalCallback`). This callback requires the environment to be wrapped
    around a `stable_baselines.bench.Monitor` wrapper to generate the 
    monitoring files that are then loaded using the 
    `stable_baselines.results_plotter.load_results` method.  
    This callback also tests the environment every `check_freq` 
    using deterministic=True. Useful to ensure that the agent is learning 
    properly. 

    '''
    
    def __init__(self, env=None, check_freq=1000, save_freq=10000, 
                 log_dir='agents', verbose=1, test=False):
        '''
        Constructor for the callback. 
        
        Parameters
        ----------
        env: BoptestGymEnv
            Environment passed here to perform tests
        check_freq: integer, default is 1000
            Number of steps to perform check and test
                check_freq: integer, default is 1000
        save_freq: integer, default is 10000
            Number of steps to store model independently of
            performance
        log_dir: string, default is 'agents'
            Path to the folder where the model will be saved. 
            It must contain the file created by an 
            `stable_baselines.bench.Monitor` wrapper. 
        verbose: integer
            Verbose level for the callback
        test: boolean
            If True, the agent is tested every `check_freq` 
            with deterministic=True 
        
        '''
        super(SaveAndTestCallback, self).__init__(verbose)
        self.env = env
        self.check_freq = check_freq
        self.save_freq = save_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.test = test

    def _init_callback(self) -> None:
        '''
        Create folder if needed
        
        '''
        
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        '''
        This method will be called by the model after each call to 
        `env.step()`.
        
        Returns
        -------
        ret_bool: boolean
            If the callback returns False, training is aborted early. In 
            this case we always return `True`. 
        
        '''
        
        # Save every self.save_freq steps independently of performance
        if self.n_calls % self.save_freq == 0:
            self.model.save(os.path.join(self.log_dir, 'model_{}'.format(self.n_calls)))
        
        if self.n_calls % self.check_freq == 0:
            print('Checking agent performance...')
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 1:
                # Mean training reward over the last self.check_freq episodes
                mean_reward = np.mean(y[-self.check_freq:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

                # New best model, we save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                        self.model.save(self.save_path)

            if self.test:
                print('Testing the agent.................................')
                test_agent(self.env, self.model, self.env.start_time, 
                           self.env.max_episode_length, self.env.warmup_period, 
                           kpis_to_file=False, plot=False, log_dir=self.log_dir)   
                # Force to render if `render_episodes` is not active
                if not self.env.render_episodes:
                    self.env.render(mode='episodes')
                # Reset the environment just in case that `self.check_freq`
                # does not coincide with a terminal state
                self.env.reset() 
        
        return True

if __name__ == "__main__":
    
    # Instantiate the env    
    env = BoptestGymEnv()

    # Check the environment
    check_env(env, warn=True)
    obs, _ = env.reset()
    env.render()
    print('Observation space: {}'.format(env.observation_space))
    print('Action space: {}'.format(env.action_space))