## TODO

1.support for llama2

    1.1 llama2 ckpt convert	DONE

    1.2 llama2 model (load), tokenizer(pad), embedding clip	DONE

    1.3 data format confirm (text completion, chat completion)  DONE

    1.4 dataset process logic DONE

    1.5 hh-rlhf dataset process DONE

    1.6 debug DONE

    1.7 first version result DONE

    1.8 eval first version result DONE

    1.9 iteration TODO

    1.9.1 Fix left pad questions

    1.9.2 add max_seq_len, max_gen_len, max_prompt_len

2.multi turns dialogs

3.support HH eval

4.support benchmark eval

5.RL

2023.08.17

1. FIx deepspeed save and reload bug

   Some weights of LlamaForCausalLM were not initialized from the model checkpoint at /mnt/petrelfs/wangxiao/AI4SocialBad/SFT/13B_3epochs_runs and are newly initialized

   utils.py line 176
2. Modify right pad -> left pad

   add max_seq_len, max_gen_len, max_prompt_len
3. DeepSpeed inference bug fix

   only output one of GPU output results
4. inference code reconstruct
5. raw_datasets.py  data load optimize		DONE

Tips

1. Llama tokenizer  自动加 bos 和 eos，不需要做额外处理
2. /training/utils/data/raw_datasets.py  里面数据保存路径需要修改
3. 在pj集群上通过vscode连接的时候，会卡在scp  vscode-server.tar 到远程这一步，解决方案：https://blog.csdn.net/m0_38040006/article/details/126752751
