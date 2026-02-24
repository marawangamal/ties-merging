echo "lambda,Avg.,qasc,wiki_qa,quartz,paws,winogrande,wsc"; \
for l in $(seq 0.1 0.1 1.0); do \
  f="exp_out/evaluation/t5-base/opm_mix_add_default/qasc,wiki_qa,quartz,paws,winogrande,wsc/lambda_${l}/inference_scores.txt"; \
  printf "%.1f," "$l"; \
  [ -f "$f" ] && tail -n1 "$f" || echo "NA"; \
done
