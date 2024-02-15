from extract import CharExtraction, SaveEmbedding, Evaluation
from config import Config
import openai

cfg = Config()

eva = Evaluation(cfg)
#eva.copy_folders_labelled()
#import os
#eva.evaluate_all_baselines()
#eva.get_all_stats()
#eva.count_character_number_all()

ext = CharExtraction(cfg)
#ext.extract_all()
#ext.extract_all(language="english")
#ext.translate_all()
#ext.check_relation_language()
ext.check_relation_language(language="english")




"""mer = SaveEmbedding(cfg)
memory_index = mer.init_memory("")
prompt = ""
memory_index.get_relevant(prompt, 10)
print(cfg.get_azure_deployment_id_for_model(cfg.embedding_model))
response = openai.Embedding.create(
    input="test",
    engine=cfg.get_azure_deployment_id_for_model(cfg.embedding_model)
)
print(response)

#src_a = os.path.join(os.getcwd(), "data", cfg.model, "relation_category_separate")
#src_b = os.path.join(os.getcwd(), "data", cfg.model, "relation_category_alltogether")
#dest_c = os.path.join(os.getcwd(), "data", cfg.model, "relation_category")
#eva.copy_folders_and_files(src_a, src_b, dest_c)

#src_a = os.path.join(os.getcwd(), "data", cfg.model, "relation_graph_separate")
#src_b = os.path.join(os.getcwd(), "data", cfg.model, "relation_graph_alltogether")
#dest_c = os.path.join(os.getcwd(), "data", cfg.model, "relation_graph")
#eva.copy_folders_and_files(src_a, src_b, dest_c)

#eva.print_label_stats(eva.all_data_dir)

"""

