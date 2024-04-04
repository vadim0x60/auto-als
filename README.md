![cover](https://user-images.githubusercontent.com/3543310/220397756-4f84d742-6ff7-44dd-86e8-349289fda79b.png)

Auto-ALS is a tool for [anthropodidactic reinforcement learning](https://vadim.me/posts/anthropodidactic/) in Healthcare based on [Virtu-ALS](https://pure.ulster.ac.uk/en/publications/ai-to-enhance-interactive-simulation-based-training-in-resuscitat) - an educational software package designed for training junior healthcare professionals in emergency care. Auto-ALS adds 
- a programmatic mode to Virtu-ALS allowing you to control the emergency care decisions in the simulator from Python via an [OpenAI Gym](https://gym.openai.com/) interface
- a hybrid mode where you make decisions manually, but a Python program is available to ask for advice or take over the controls at anytime

Note that the Auto-ALS only provides an interface to connect a decision-making algorithm to Virtu-ALS, but not the algorithm itself, though see `examples` for a sketch of such algorithm.

For more context see our paper, [Towards Effective Patient Simulators](https://www.frontiersin.org/articles/10.3389/frai.2021.798659/full). If you use Auto-ALS in research, please cite it:

```
@ARTICLE{10.3389/frai.2021.798659,
AUTHOR={Liventsev, Vadim and Härmä, Aki and Petković, Milan},   
TITLE={Towards Effective Patient Simulators},      
JOURNAL={Frontiers in Artificial Intelligence},      
VOLUME={4},      
YEAR={2021},      
URL={https://www.frontiersin.org/article/10.3389/frai.2021.798659},       
DOI={10.3389/frai.2021.798659},      
ISSN={2624-8212},   
ABSTRACT={In this paper we give an overview of the field of patient simulators and provide qualitative and quantitative comparison of different modeling and simulation approaches. Simulators can be used to train human caregivers but also to develop and optimize algorithms for clinical decision support applications and test and validate interventions. In this paper we introduce three novel patient simulators with different levels of representational accuracy: HeartPole, a simplistic transparent rule-based system, GraphSim, a graph-based model trained on intensive care data, and Auto-ALS—an adjusted version of an educational software package used for training junior healthcare professionals. We provide a qualitative and quantitative comparison of the previously existing as well as proposed simulators.}
}
```

## Programmatic mode

```
import gym
import auto_als

# Uncomment whether you want to render the environment. It will let you follow exactly what the agent is doing in a 3D simulation, but the non-rendered version is considerably faster:
# render = True
# render = False

env = gym.make('Auto-ALS-v0', render=render)
observation = env.reset()
done = False
while not done:
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()
env.close()
```

With this, your agent will keep applying random medical procedures to John until John gets better (unlikely), dies or the agent selects action 36 (Finish) and gives up on John. If you set `render=True` you will get to watch it happen in a 3D simulation:

![Poor John](https://static.vadim.me/Virtu-ALS.png)

## Hybrid mode

To start Auto-ALS in hybrid mode set `autoplay` parameter to `False`

```
env = gym.make('Auto-ALS-v0', render=True, autoplay=False)
```

The environment will start in an interactive mode and this time you will be able to decide what's in store for John, by pressing action buttons on the left side of the screen. However, you also have the AI menu available:

![AI Menu](https://static.vadim.me/aimenu.png)

Your python script is currently paused. But if you press the help (left) button  or the decision (middle) button, the script will be unpaused until it decides on an `action` and passes it to `env.step()`. If you pressed the decision button, this `action` will be implemented and if you pressed the help button it will be indicated to you, but not implemented - it's up to you to actually do it. The right (play/pause) button turns on full autopilot, unpauses your python script and doesn't pause it again until you pause it manually with the pause button.

So, `action` and `observation` variables are used to establish communication between the control script and the simulator. But what are these variable exactly?

## Observation space

The `observation` is a numpy vector of size 33+7+7=47 that represents various events that happened in the environment prior to the moment when `observation` is received. It implements an event encoding strategy described in [Reinforcement Learning as Message Passing](https://vadim.me/posts/mpdp/).

The first 33 elements indicate whether one of 33 types of events has occured and how long ago it happened. For i=1,2,...,33:

$$o_i = \exp(t_i - t)$$

i.e. the inverse exponent of how much time has passed since this event has last occured that can be interpreted as _relevance_ of this event at the moment. If the event has never occured,  

$$t - t_i = \infty \implies \exp(t_i - t) = 0$$

The 33 events are as follows: 

```
    ResponseVerbal,
    ResponseGroan,
    ResponseNone,
    AirwayClear,
    AirwayVomit,
    AirwayBlood,
    AirwayTongue,
    BreathingNone,
    BreathingSnoring,
    BreathingSeeSaw,
    BreathingEqualChestExpansion,
    BreathingBibasalCrepitations,
    BreathingWheeze,
    BreathingCoarseCrepitationsAtBase,
    BreathingPneumothoraxSymptoms,
    VentilationResistance,
    RadialPulsePalpable,
    RadialPulseNonPalpable,
    HeartSoundsMuffled,
    HeartSoundsNormal,
    AVPU_A,
    AVPU_U,
    AVPU_V,
    PupilsPinpoint,
    PupilsNormal,
    ExposureRash,
    ExposurePeripherallyShutdown,
    ExposureStainedUnderwear,
    HeartRhythm0,
    HeartRhythm1,
    HeartRhythm2,
    HeartRhythm3,
    HeartRhythm4
```

The next 7 components use the same time encoding $o_i = \exp(t_i - t)$ for vital signs measurement, i.e. how recently the last measurement has occured:

```
    MeasuredHeartRate,
    MeasuredRespRate,
    MeasuredCapillaryGlucose,
    MeasuredTemperature,
    MeasuredMAP,
    MeasuredSats,
    MeasuredResps
```

The last 7 components contain the measurements themselves.

The above list of observation labels can be accessed with

```
from auto_als import observations
```

## Action space

`action` should be an integer no less than 0 and no more than 36. The 37 actions are:

```
    DoNothing,
    ABG,
    AirwayManoeuvres,
    GiveAtropine,
    GiveAdenosine,
    GiveAdrenaline,
    GiveAmiodarone,
    GiveMidazolam,
    Venflon,
    Yankeur,
    DrawBloods,
    BPCuffOn,
    BVM,
    Guedel,
    NRBMask,
    DefibOn,
    DefibAttachPads ,
    DefibShock,
    DefibCharge ,
    DefibChangePaceCurrentDown,
    DefibChangePaceCurrent,
    DefibEnergyDown,
    DefibEnergyUp,
    DefibChangePaceRateDown,
    DefibChangePaceRateUp,
    DefibPace,
    DefibPacePause,
    DefibSync,
    AssessResponse
    AssessAirway
    AssessBreathing,
    AssessCirculation,
    AssessDisability,
    AssessExposure,
    AssessDefibrillator,
    AssessMonitor
    Finish
```

Note, in particular, the `Assess` actions. These actions, just like `DoNothing` are guaranteed to have no effect on the patient state. However, some observation events will not trigger unless you go looking for them. To check the blood pressure, one needs to attach the blood pressure cuff to the patient and look at the monitor. Hence, the `MeasuredMAP` event will only trigger after you `BPCuffOn` and `AssessMonitor`. [Assessment skills](https://www.resus.org.uk/library/abcde-approach) (knowing where to look and how to establish the patient's state) are crucial for patient resusciation - the simulation would be woefully inadequate if the assessments were just provided for you automatically.

The above list of actions can be accessed with

```
from auto_als import actions
```

Good luck!
