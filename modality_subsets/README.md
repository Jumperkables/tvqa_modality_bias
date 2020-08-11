# Modality Subsets

The percentage of questions answered correctly by Group A and also not answered correctly by Group B. Where Group B is null '-', it just corresponds to questions answered by Group A. We present splits of the training and validation sets derived from our BERT and GloVe model versions. We note that proposing subsets of the training set is somewhat limited because our models are trained on it. However we still believe these splits are worth sharing.
<p align="center">
  <img src="https://github.com/Jumperkables/kable_management/blob/master/tvqa_modality_bias/modality_subsets/iem_table.png" alt="IEM Table"/>
</p>

Questions are identified by their question id with respect to the original TVQA splits.

-<em>All:</em> All models <br>
-<em>Subtitle</em>: Only models trained <strong>with</strong> subtitles<br>
-<em>Non-Subtitle:</em> Only models trained <strong>without</strong> subtitles<br>
-<em>Singleton:</em> All models trained on only a single features i.e. <em>S, V, I, R</em><br>
-We name the split of questions that are not answered correctly by any of our models as the <em>Hard</em> split 
