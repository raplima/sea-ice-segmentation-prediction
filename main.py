import argparse
import configparser
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rioxarray
import torch
import xarray as xr
from torch import nn
from torchvision.transforms import Normalize
from tqdm import tqdm


def write_raster(out:torch.Tensor, 
                 out_fname:str, 
                 raster_example:xr.core.dataarray.DataArray, 
                 dtype:str, 
                 nodata:Optional[int]=None):
    """Writes a raster to disk.

    Args:
        out (torch.Tensor): Array output to be written into raster
        out_fname (str): Raster output filename
        raster_example (xr.core.dataarray.DataArray): A raster template that contains the same projection information desired 
                                                      for the output raster. 
        dtype (str): Data type for the output raster
        nodata (Optional[int], optional): uint8 rasters need to set nodata values as 241. Defaults to None.
    """
    xr_res = xr.DataArray(out, 
                            [('band', np.arange(1, out.shape[0]+1)),
                            ('y', raster_example.y.values),
                            ('x', raster_example.x.values)])
    
    xr_res['spatial_ref']=raster_example.spatial_ref                              
    xr_res.attrs=raster_example.attrs
    
    if nodata:
        xr_res.rio.write_nodata(nodata, inplace=True)

    # write to file
    if os.path.isfile(out_fname):
        os.remove(out_fname)
    xr_res.rio.to_raster(out_fname, dtype=dtype)


def main(config:configparser.ConfigParser):
    """Main function. Takes in a configuration information

    Args:
        config (configparser.ConfigParser): Configuration object containing:
            output directory
            full filepaths for trained models
            full filepaths for input rasters
    """

    start_time = time.perf_counter()

    # output directory
    dir_out = os.path.normpath(config['io']['dir_out'])

    # read in raster filenames
    input_rasters = [os.path.join(f) for f in config['io']['input_rasters'].split('\n')]    

    # user can define one or more models
    model_paths = [os.path.join(f) for f in config['io']['model_path'].split('\n')]

    # device
    device = torch.device(config['io']['device'])
    
    # read in one or more models
    models = []
    for model_path in model_paths:
        models.append(torch.jit.load(model_path))
    # make sure they are in evaluation mode
    for model in models:
        model.eval()
        model.to(device)

    # models were trained assuming mean and standard deviation provided in 
    # https://github.com/astokholm/AI4ArcticSeaIceChallenge/tree/main/misc
    input_features = ['nersc_sar_primary',
                      'nersc_sar_secondary',
                      'sar_incidenceangle']
    
    global_meanstd = np.load(os.path.join('misc', 'global_meanstd.npy'), allow_pickle=True).item()
    
    mean = [global_meanstd[val]['mean'] for val in input_features]
    std =  [global_meanstd[val]['std'] for val in input_features]
    norms = Normalize(mean, std)

    # save configuration file:
    if not os.path.isdir(dir_out):
        os.mkdir(dir_out)
    with open(os.path.join(dir_out, f'config.cfg'), 'w') as out_file:
        config.write(out_file)

    # run on test rasters:
    softmax = nn.Softmax(0)
    for input_raster in tqdm(input_rasters):

        raster = rioxarray.open_rasterio(input_raster, masked=True)
        x = torch.from_numpy(raster.values).unsqueeze(dim=0)

        # get input mask 
        mask = np.isnan(raster.values).any(axis=0)

        # normalize
        x = torch.nan_to_num(norms(x))

        res = []
        for model in models:
            with torch.no_grad():
                res.append(softmax(torch.squeeze(model(x.to(device)).detach().cpu(),0)))
        
        # calculate mean and std and mark nan vals:
        res_mean =  torch.mean(torch.stack(res), dim=0)
        for band in res_mean:
            band[mask] = np.nan

        if len(models) > 1:
            res_std =  torch.std(torch.stack(res), dim=0)
            for band in res_std:
                band[mask] = np.nan

        #####################################################
        # write output rasters
        #####################################################
        ###### mean ice probability
        out_fname = os.path.join(dir_out, f'mean-prob-{Path(input_raster).stem}.tif')
        write_raster(res_mean, out_fname, raster, dtype='float32')

        ##### classification
        y_pred_class = res_mean.argmax(0)
        # 241 is the no data value for uint8
        nodata = 241
        y_pred_class[mask] = nodata
        y_pred_class = np.expand_dims(y_pred_class, 0)

        out_fname = os.path.join(dir_out, f'class-{Path(input_raster).stem}.tif')
        write_raster(y_pred_class, out_fname, raster, dtype='uint8', nodata=nodata)


        ###### std
        if len(models) > 1:
            out_fname = os.path.join(dir_out, f'pred-std-{Path(input_raster).stem}.tif')
            write_raster(res_std, out_fname, raster, dtype='float32')
    
    end_time = time.perf_counter()
    print(f'Program terminated successfully in {(end_time-start_time)/60:.2f} minutes')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config_file', default='config.cfg')

    args = parser.parse_args()

    if os.path.isfile(args.config_file):
        config = configparser.ConfigParser()
        config.read(args.config_file)

        main(config)
    
    else:
        print('Please provide a valid configuration file.')
