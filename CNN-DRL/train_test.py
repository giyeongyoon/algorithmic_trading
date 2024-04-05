from meta.data_processor import DataProcessor
from envs.StockTradingEnvCandle import StockTradingEnvCandle, get_candlestick
from agents2.agent import DRLAgent
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3

MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}

def train(data_source, start_date, end_date, time_interval,
          ticker_list, technical_indicator_list, env, model_name,
          if_vix=True, cache=False, select_stockstats_talib=0,
          hmax=100, initial_amount=1000000, reward_scaling=1e-4,
          transaction_cost_pct=0.001, **kwargs):

    # fetch data
    import warnings
    warnings.filterwarnings("ignore")

    DP = DataProcessor(
        data_source=data_source,
        start_date=start_date,
        end_date=end_date,
        time_interval=time_interval
    )
    price_array, tech_array, turbulence_array = DP.run(
        ticker_list=ticker_list,
        technical_indicator_list=technical_indicator_list,
        if_vix=if_vix,
        cache=cache,
        select_stockstats_talib=select_stockstats_talib
    )

    df = DP.dataframe
    df = df.sort_values(['time', 'tic'], ignore_index=True)

    data_dir = kwargs.get('data_dir', "")

    if kwargs.get('generate_candlestick', False):
        term = kwargs.get('term', 28)
        for i in range(term, df.shape[0] + 1):
            get_candlestick(df, start=i, term=term, data_dir=data_dir)

    stock_dimension = len(df.tic.unique())
    state_space = 1 + 2*stock_dimension + len(technical_indicator_list)*stock_dimension

    buy_cost_list = sell_cost_list = [transaction_cost_pct] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": hmax,
        "initial_amount": initial_amount,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": technical_indicator_list,
        "action_space": stock_dimension,
        "reward_scaling": reward_scaling
    }

    if env == StockTradingEnvCandle:
        env_kwargs["data_dir"] = data_dir
    else:
        pass

    env_instance = env(df=df, **env_kwargs)

    cwd = kwargs.get('cwd','./'+str(model_name))

    total_timesteps = kwargs.get('total_timesteps', 1e6)
    agent_params = kwargs.get('agent_params')
    policy_kwargs = kwargs.get('policy_kwargs', None)
    policy = kwargs.get('policy', 'MlpPolicy')
    trained_model = kwargs.get('trained_model', None)

    agent = DRLAgent(env=env_instance)

    model = agent.get_model(model_name, policy=policy, model_kwargs=agent_params, policy_kwargs=policy_kwargs, trained_model=trained_model)
    trained_model = agent.train_model(model=model,
                            tb_log_name=model_name,
                            total_timesteps=total_timesteps)
    print('Training finished!')
    trained_model.save(cwd)
    print('Trained model saved in ' + str(cwd))

def test(data_source, start_date, end_date, time_interval,
         ticker_list, technical_indicator_list, env, model_name,
         if_vix=True, cache=False, select_stockstats_talib=0,
         hmax=100, initial_amount=1000000, reward_scaling=1e-4,
         transaction_cost_pct=0.001, **kwargs):
    
    # fetch data
    import warnings
    warnings.filterwarnings("ignore")

    DP = DataProcessor(
        data_source=data_source,
        start_date=start_date,
        end_date=end_date,
        time_interval=time_interval
    )
    price_array, tech_array, turbulence_array = DP.run(
        ticker_list=ticker_list,
        technical_indicator_list=technical_indicator_list,
        if_vix=if_vix,
        cache=cache,
        select_stockstats_talib=select_stockstats_talib
    )

    df = DP.dataframe
    df = df.sort_values(['time', 'tic'], ignore_index=True)

    data_dir = kwargs.get('data_dir', "")

    if kwargs.get('generate_candlestick', False):
        term = kwargs.get('term', 28)
        for i in range(term, df.shape[0] + 1):
            get_candlestick(df, start=i, term=term, data_dir=data_dir)

    stock_dimension = len(df.tic.unique())
    state_space = 1 + 2*stock_dimension + len(technical_indicator_list)*stock_dimension

    buy_cost_list = sell_cost_list = [transaction_cost_pct] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": hmax,
        "initial_amount": initial_amount,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": technical_indicator_list,
        "action_space": stock_dimension,
        "reward_scaling": reward_scaling
    }

    if env == StockTradingEnvCandle:
        env_kwargs["data_dir"] = data_dir
    else:
        pass

    env_instance = env(df=df, **env_kwargs)

    cwd = kwargs.get('cwd','./'+str(model_name))

    trained_model = MODELS[model_name].load(cwd, env=env_instance)

    df_daily_return, df_actions = DRLAgent.DRL_prediction(
        model=trained_model,
        environment=env_instance,

    )
    df_daily_return["method"] = model_name
    return df_daily_return, df_actions