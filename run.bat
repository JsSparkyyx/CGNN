for %%l in (1,2,3,4,5) do python main.py --method CFD --arch HTG --dataset cfd --manner full_batch --seed %%l
for %%l in (1,2,3,4,5) do python main.py --method EWC --arch HTG --dataset cfd --manner full_batch --seed %%l
for %%l in (1,2,3,4,5) do python main.py --method Finetune --arch HTG --dataset cfd --manner full_batch --seed %%l
