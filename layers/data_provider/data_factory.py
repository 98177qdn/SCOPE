from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, \
    Dataset_Custom_No_Target_Feature, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, Dataset_Custom_no_dirty_data,Dataset_Custom_freq_data,Dataset_Custom_61to32_39Hzs,Dataset_Custom_30_to_32_series_fit,\
    Dataset_Custom_enc_in_forcast_c_out,Dataset_Custom_enc_in_forcast_enc_in,Dataset_Custom_enc_in_fit_c_out
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'custom_no_target_feature': Dataset_Custom_No_Target_Feature,
    'custom_no_dirty': Dataset_Custom_no_dirty_data,
    'custom_freq_data': Dataset_Custom_freq_data,
    'custom_61to32_30Hzs': Dataset_Custom_61to32_39Hzs,
    'custom_30to32_series_fit':Dataset_Custom_30_to_32_series_fit,
    'custom_enc_in_forcast_c_out':Dataset_Custom_enc_in_forcast_c_out,
    'custom_enc_in_forcast_enc_in':Dataset_Custom_enc_in_forcast_enc_in,
    'custom_enc_in_fit_c_out':Dataset_Custom_enc_in_fit_c_out,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args=args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args=args,
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            args=args,
            scale=args.scale,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.x_len, args.label_len, args.y_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
