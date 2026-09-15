import numpy as np

np.random.seed(0)

from models.deform import *
from models.deform_unet import *
from models.pgsfi_net import PGSFINet
from models.pgsfi_net_sobel_only import PGSFINetSobelOnly
from models.pgsfi_net_spatial_only import PGSFINetSpatialOnly
from models.pgsfi_net_frequency_only import PGSFINetFrequencyOnly
from models.pgsfi_net_concat import PGSFINetConcat
from models.pgsfi_net_addition import PGSFINetAddition
from models.pgsfi_net_grid_only import PGSFINetGridOnly
from models.pgsfi_net_block_only import PGSFINetBlockOnly
from models.pgsfi_net_local_only import PGSFINetLocalOnly
from models.pgsfi_net_global_only import PGSFINetGlobalOnly
from models.pgsfi_net_sap import PGSFINetSAP
from models.pgsfi_net_sap_fbsaa import PGSFINetSAPFBSAA
from models.pgsfi_net_sap_fbsaa_mbsff import PGSFINetSAPFBSAAMBSFF
from models.pgsfi_net_no_prior import PGSFINetNoPrior
from models.pgsfi_net_no_fbsaa import PGSFINetNoFbsaa
from models.pgsfi_net_no_mbsff import PGSFINetNoMbsff
from models.pgsfi_net_no_decoder import PGSFINetNoDecoder
from models.unet import UNet
from models.spatial_freq_domain import (
    SpatialBranch, 
    FrequencyBranch, 
    SpatialFreqDomainBlock, 
    MultiBranchSpatialFreq, 
    VesselPriorExtractor,
    Plain_Four_SAFM_Block,
    FourBranchModule,
    SAFM,
    StripConv,
    SpatialAttention
)

MODELS = {'unet': UNet,
          'deform_v1': DeformConvNetV1V2,
          'deform_unet_v1': DeformUNetV1V2,
          'pgsfi_net': PGSFINet,
          'pgsfi_net_sobel_only': PGSFINetSobelOnly,
          'pgsfi_net_spatial_only': PGSFINetSpatialOnly,
          'pgsfi_net_frequency_only': PGSFINetFrequencyOnly,
          'pgsfi_net_concat': PGSFINetConcat,
          'pgsfi_net_addition': PGSFINetAddition,
          'pgsfi_net_grid_only': PGSFINetGridOnly,
          'pgsfi_net_block_only': PGSFINetBlockOnly,
          'pgsfi_net_local_only': PGSFINetLocalOnly,
          'pgsfi_net_global_only': PGSFINetGlobalOnly,
          'pgsfi_net_sap': PGSFINetSAP,
          'pgsfi_net_sap_fbsaa': PGSFINetSAPFBSAA,
          'pgsfi_net_sap_fbsaa_mbsff': PGSFINetSAPFBSAAMBSFF,
          'pgsfi_net_no_prior': PGSFINetNoPrior,
          'pgsfi_net_no_fbsaa': PGSFINetNoFbsaa,
          'pgsfi_net_no_mbsff': PGSFINetNoMbsff,
          'pgsfi_net_no_decoder': PGSFINetNoDecoder,
          }
