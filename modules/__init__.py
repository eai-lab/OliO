
def get_wrapper(args, backbone):
    if not args.module_version == 'timesiam':
        if args.projection_dim < args.max_tau:
            raise ValueError("projection_dim must be greater than or equal to max_tau")
    
    
    if args.module_version == 'v1-triangular-causal':
        from .oilo_v1_triangular_causal import OilO_Wrapper
        print("Using OilO_Wrapper: v1-triangular-causal")
        return OilO_Wrapper(args, backbone)
    elif args.module_version == 'v1-triangular-causal-channel-individual':
        if args.fast_ci_olio:
            from .oilo_v1_triangular_causal_channel_individual_fast import OilO_Wrapper
        else:
            from .oilo_v1_triangular_causal_channel_individual import OilO_Wrapper
        print(f"Using OilO_Wrapper: v1-triangular-causal-channel-individual fast option: {args.fast_ci_olio}")
        return OilO_Wrapper(args, backbone)

    raise ValueError(f"Unknown module version: {args.module_version}")