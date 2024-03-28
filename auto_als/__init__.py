from gymnasium.envs.registration import register

from auto_als.envs import AutoALSException

register(
    id='Auto-ALS-v0',
    entry_point='auto_als.envs:AutoALS',
    nondeterministic=True
)

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
    'DefibPacePause',
    'DefibSync',
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
    'HeartRateKnown',
    'RespRateKnown',
    'CapillaryGlucoseKnown',
    'TemperatureKnown',
    'MAPKnown',
    'SatsKnown',
    'RespsKnown',
    'HeartRate',
    'RespRate',
    'CapillaryGlucose',
    'Temperature',
    'MAP',
    'Sats',
    'Resps'
]