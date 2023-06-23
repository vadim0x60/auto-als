import gym
import auto_als

actions = [
    'DoNothing',
    'ABG',
    'AirwayManoeuvres',
    'GiveAtropine',
    'GiveAdenosine',
    'GiveAdrenaline',
    'GiveAmiodarone',
    'GiveMidazolam',
    'Venflon',
    'Yankeur',
    'DrawBloods',
    'BPCuffOn',
    'BVM',
    'Guedel',
    'NRBMask',
    'DefibOn',
    'DefibAttachPads ',
    'DefibShock',
    'DefibCharge ',
    'DefibChangePaceCurrentDown',
    'DefibChangePaceCurrent',
    'DefibEnergyDown',
    'DefibEnergyUp',
    'DefibChangePaceRateDown',
    'DefibChangePaceRateUp',
    'DefibPace',
    'AssessResponse',
    'AssessAirway',
    'AssessBreathing',
    'AssessCirculation',
    'AssessDisability',
    'AssessExposure',
    'AssessDefibrillator',
    'AssessMonitor',
    'Finish'
    ]

observations = [
    'Blunder',
    'Success',
    'Failure',
    'ResponseVerbal',
    'ResponseGroan',
    'ResponseNone',
    'AirwayClear',
    'AirwayVomit',
    'AirwayBlood',
    'AirwayTongue',
    'BreathingNone',
    'BreathingSnoring',
    'BreathingSeeSaw',
    'BreathingEqualChestExpansion',
    'BreathingBibasalCrepitations',
    'BreathingWheeze',
    'BreathingCoarseCrepitationsAtBase',
    'BreathingPneumothoraxSymptoms',
    'VentilationResistance',
    'RadialPulsePalpable',
    'RadialPulseNonPalpable',
    'HeartSoundsMuffled',
    'HeartSoundsNormal',
    'AVPU_A',
    'AVPU_U',
    'AVPU_V',
    'PupilsPinpoint',
    'PupilsNormal',
    'ExposureRash',
    'ExposurePeripherallyShutdown',
    'ExposureStainedUnderwear',
    'HeartRhythm0',
    'HeartRhythm1',
    'HeartRhythm2',
    'HeartRhythm3',
    'HeartRhythm4',
    'MeasuredHeartRate',
    'MeasuredRespRate',
    'MeasuredCapillaryGlucose',
    'MeasuredTemperature',
    'MeasuredMAP',
    'MeasuredSats',
    'MeasuredResps'
]

actions = {a: i for i, a in enumerate(actions)}

env = gym.make('Auto-ALS-v0', attach=False, render=True, autoplay=False)
env.reset()
env.step(actions['AssessAirway'])
env.step(actions['AssessBreathing'])
env.step(actions['AssessCirculation'])
env.step(actions['AssessDisability'])
env.step(actions['AssessExposure'])