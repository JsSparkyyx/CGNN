@REM for %%l in (1,2,3,4,5) do python main.py --method EWC --arch GAT --dataset cora --manner full_batch --seed %%l
@REM for %%l in (1,2,3,4,5) do python main.py --method EWC --arch GAT --dataset reddit --manner full_batch --seed %%l

@REM for %%l in (1,2,3,4,5) do python main.py --method EWC --arch GCN --dataset cora --manner full_batch --seed %%l
@REM for %%l in (1,2,3,4,5) do python main.py --method EWC --arch GCN --dataset reddit --manner full_batch --seed %%l

@REM for %%l in (1,2,3,4,5) do python main.py --method EWC --arch SAGE --dataset cora --manner full_batch --seed %%l
@REM for %%l in (1,2,3,4,5) do python main.py --method EWC --arch SAGE --dataset reddit --manner full_batch --seed %%l

@REM for %%l in (1,2,3,4,5) do python main.py --method MAS --arch GAT --dataset cora --manner full_batch --seed %%l
@REM for %%l in (1,2,3,4,5) do python main.py --method MAS --arch GAT --dataset reddit --manner full_batch --seed %%l

@REM for %%l in (1,2,3,4,5) do python main.py --method MAS --arch GCN --dataset cora --manner full_batch --seed %%l
@REM for %%l in (1,2,3,4,5) do python main.py --method MAS --arch GCN --dataset reddit --manner full_batch --seed %%l

@REM for %%l in (1,2,3,4,5) do python main.py --method MAS --arch SAGE --dataset cora --manner full_batch --seed %%l
@REM for %%l in (1,2,3,4,5) do python main.py --method MAS --arch SAGE --dataset reddit --manner full_batch --seed %%l

for %%l in (1,2,3,4,5) do python main.py --method GEM --arch GAT --dataset cora --manner full_batch --seed %%l
for %%l in (1,2,3,4,5) do python main.py --method GEM --arch GAT --dataset reddit --manner full_batch --seed %%l

for %%l in (1,2,3,4,5) do python main.py --method GEM --arch GCN --dataset cora --manner full_batch --seed %%l
for %%l in (1,2,3,4,5) do python main.py --method GEM --arch GCN --dataset reddit --manner full_batch --seed %%l

for %%l in (1,2,3,4,5) do python main.py --method GEM --arch SAGE --dataset cora --manner full_batch --seed %%l
for %%l in (1,2,3,4,5) do python main.py --method GEM --arch SAGE --dataset reddit --manner full_batch --seed %%l