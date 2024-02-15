import json, os
import yaml
#from memory.processing import split_text
from config.utils.token_counter import count_string_tokens
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class Evaluation():
    def __init__(self, config):
        self.config = config
        self.retry = True
        prompt_path = os.path.join(os.getcwd(), "prompts.yaml")
        with open(prompt_path, 'r') as f:
            self.prompts = yaml.load(f, Loader=yaml.FullLoader)
            f.close()
        #self.category_list = self.prompts['relation_list'].split("\n")

        category_path = os.path.join(os.getcwd(), "data", "equivalent_relation.json")
        with open(category_path, 'r') as f:
            self.equivalent_relation = json.load(f)
            f.close()
        self.high_level_relations = self.equivalent_relation["High-Level relations"]
        self.high_level_relations = {k.lower().strip().replace("-"," "): v for k, v in self.high_level_relations.items()}

        self.high_level_category_map = {}
        for relation in self.high_level_relations:
            for mapping_relation in self.high_level_relations[relation]:
                relation = relation.lower().strip().replace("-"," ")
                mapping_relation = mapping_relation.lower().strip().replace("-"," ")
                self.high_level_category_map[mapping_relation] = relation

        self.equivalent_relation = self.equivalent_relation["equivalent relations"]
        self.equivalent_relation = {k.lower().strip().replace("-"," "): v for k, v in self.equivalent_relation.items()}
        self.relation_merge_map = {}
        for relation in self.equivalent_relation:
            for mapping_relation in self.equivalent_relation[relation]:
                relation = relation.lower().strip().replace("-"," ")
                mapping_relation = mapping_relation.lower().strip().replace("-"," ")
                self.relation_merge_map[mapping_relation] = relation
        
        self.category_list = list(self.equivalent_relation.keys()) + list(self.relation_merge_map.keys())
        self.category_list = [item.lower().strip().replace("-"," ") for item in self.category_list]
        print("type of total categories", len(self.category_list), self.category_list)
        #print("type of total merged categories", self.equivalent_relation.keys())

        language = "chinese"
        self.all_data_dir = {}
        #self.all_data_dir["gpt-3.5-turbo-whole-graph"] = os.path.join(os.getcwd(), "data", language, "gpt-3.5-turbo", "extract_whole_graph")
        #self.all_data_dir["gpt-3.5-turbo-direct-ask"] = os.path.join(os.getcwd(), "data", language, "gpt-3.5-turbo", "relation_extract_directly")
        #self.all_data_dir["gpt-3.5-turbo-pairwise-ask"] = os.path.join(os.getcwd(), "data", language, "gpt-3.5-turbo", "relation_extract_pairwisely", "given")
        #self.all_data_dir["gpt-3.5-turbo-pairwise-ask"] = os.path.join(os.getcwd(), "data", language, "gpt-3.5-turbo", "relation_extract_pairwisely", "given")
        #self.all_data_dir["gpt-4-turbo-direct-ask"] = os.path.join(os.getcwd(), "data", "chinese", "gpt-4", "relation_category")
        #self.all_data_dir["gpt-4-turbo-pairwise-ask"] = os.path.join(os.getcwd(), "data", "chinese", "gpt-4", "relation_pairwise_category")
        
        self.label_dir = os.path.join(os.getcwd(), "data", language, "label")
        self.annotation_dir = os.path.join(os.getcwd(), "data", "label_agreement")

        self.input_narrative = os.path.join(os.getcwd(), "data", language, "data_final")
        self.stats_path = os.path.join(os.getcwd(), "data", "stats.json")
        self.evaluation_results_path = os.path.join(os.getcwd(), "data", "evaluation_results.json")
    
    """def evaluate_all_baselines(self):
        stats_to_save = {}
        for check_type, check_dir in self.all_data_dir.items():
            evaluation_level_results = {}
            for evaluation_level in range(3):
                self.evaluation_level = evaluation_level
                temp_evaluation_results = self.cal_f1(check_dir, self.label_dir, evaluate_type="merged")
                temp_evaluation_results = self.cal_f1(check_dir, self.label_dir, evaluate_type="personal", results_to_save = temp_evaluation_results)
                temp_evaluation_results = self.cal_f1(check_dir, self.label_dir, evaluate_type="all", results_to_save = temp_evaluation_results)
                evaluation_level_results[evaluation_level] = temp_evaluation_results
            temp_evaluation_results = self.cal_sim(check_dir, self.label_dir, evaluate_type="merged")
            temp_evaluation_results = self.cal_sim(check_dir, self.label_dir, evaluate_type="personal", results_to_save = temp_evaluation_results)
            temp_evaluation_results = self.cal_sim(check_dir, self.label_dir, evaluate_type="all", results_to_save = temp_evaluation_results)
            evaluation_level_results["relation similarity score"] = temp_evaluation_results 
            stats_to_save[check_type] = evaluation_level_results

        #stats_to_save["llm generated"] 
        with open(self.evaluation_results_path, 'w') as f:
            json.dump(stats_to_save, f, ensure_ascii=False, indent=4)
            f.close()

    def cal_f1(self, generate_data_path, label_path, evaluate_type="merged", results_to_save = {}, merged_file="all.json"):
        generated_relations_all = []
        labelled_relations_all = []
        n_generated_ic_all, n_generated_ooc_all, n_labelled_ic_all, n_labelled_ooc_all = 0, 0, 0, 0
        for file in os.listdir(label_path):
            if file == ".DS_Store":
                continue
            else:
                for char_file in os.listdir(os.path.join(label_path, file)):
                    if file == ".DS_Store":
                        continue
                    else:
                        if char_file== "all.json":
                            generate_data_file_path = os.path.join(generate_data_path, file, merged_file)
                        else:
                            generate_data_file_path = os.path.join(generate_data_path, file, char_file)

                        if os.path.isfile(generate_data_file_path):
                            with open(generate_data_file_path, 'r') as f:
                                generated_data = json.load(f)
                                generated_data, _, _ = self.check_format(generated_data)
                                f.close()
                        else:
                            continue
                        with open(os.path.join(label_path, file, char_file), 'r') as f:
                            labelled_data = json.load(f)
                            labelled_data, _, _ = self.check_format(labelled_data)
                            f.close()

                    if evaluate_type == "merged":
                        if char_file == "all.json":
                            generated_relations_all, labelled_relations_all, n_generated_ic_all, n_generated_ooc_all, n_labelled_ic_all, n_labelled_ooc_all = self.compare_f1(generated_data, labelled_data, generated_relations_all, labelled_relations_all, n_generated_ic_all, n_generated_ooc_all, n_labelled_ic_all, n_labelled_ooc_all)
                    elif evaluate_type == "personal":
                        if char_file != "all.json":
                            generated_relations_all, labelled_relations_all, n_generated_ic_all, n_generated_ooc_all, n_labelled_ic_all, n_labelled_ooc_all = self.compare_f1(generated_data, labelled_data, generated_relations_all, labelled_relations_all, n_generated_ic_all, n_generated_ooc_all, n_labelled_ic_all, n_labelled_ooc_all)
                    else:
                        generated_relations_all, labelled_relations_all, n_generated_ic_all, n_generated_ooc_all, n_labelled_ic_all, n_labelled_ooc_all = self.compare_f1(generated_data, labelled_data, generated_relations_all, labelled_relations_all, n_generated_ic_all, n_generated_ooc_all, n_labelled_ic_all, n_labelled_ooc_all)
        
        if n_generated_ic_all + n_generated_ooc_all == 0:
            results_to_save[evaluate_type] = {'corruption_rate': 1, "accuracy": 'na', "precision": 'na', "recall": 'na', "f1": 'na'}
            return results_to_save
        else:
            corruption_rate = n_generated_ooc_all/(n_generated_ic_all + n_generated_ooc_all)
            accuracy = accuracy_score(labelled_relations_all, generated_relations_all)
            precision = precision_score(labelled_relations_all, generated_relations_all, average='macro')
            recall = recall_score(labelled_relations_all, generated_relations_all, average='macro')
            f1 = f1_score(labelled_relations_all, generated_relations_all, average='macro')
            results_to_save[evaluate_type] = {'corruption_rate': corruption_rate, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
            return results_to_save

    def compare_f1(self, generated_data, labelled_data, generated_relations_all, labelled_relations_all, n_generated_ic_all, n_generated_ooc_all, n_labelled_ic_all, n_labelled_ooc_all):
        #generated_relations_all = []
        #labelled_relations_all = []
        #n_generated_ic_all, n_generated_ooc_all, n_labelled_ic_all, n_labelled_ooc_all = 0, 0, 0, 0
        for character, labelled_relations in labelled_data.items():
            labelled_relations = {item[0]: item[1] for item in labelled_relations}
            if character in generated_data:
                generated_relations ={item[0]: item[1] for item in generated_data[character]}
            else:
                generated_relations ={}
            for related_character in labelled_relations:
                if related_character in generated_relations:
                    generated_relations_match, labelled_relations_match, n_generated_ic, n_generated_ooc, n_labelled_ic, n_labelled_ooc = self.match_relations(generated_relations[related_character], labelled_relations[related_character])
                else:
                    generated_relations_match, labelled_relations_match, n_generated_ic, n_generated_ooc, n_labelled_ic, n_labelled_ooc = self.match_relations("", labelled_relations[related_character])
                n_generated_ic_all += n_generated_ic
                n_generated_ooc_all += n_generated_ooc
                n_labelled_ic_all += n_labelled_ic
                n_labelled_ooc_all += n_labelled_ooc
                generated_relations_all += generated_relations_match
                labelled_relations_all += labelled_relations_match
        
        return generated_relations_all, labelled_relations_all, n_generated_ic_all, n_generated_ooc_all, n_labelled_ic_all, n_labelled_ooc_all
        
        
    def match_relations(self, generated_relations, labelled_relations):
        generated_relations, n_generated_ic, n_generated_ooc = self.remove_corrupted_relations([item.lower().strip().replace("-"," ") for item in generated_relations.split(",")])
        labelled_relations, n_labelled_ic, n_labelled_ooc = self.remove_corrupted_relations([item.lower().strip().replace("-"," ") for item in labelled_relations.split(",")])
        generated_relations = self.change_relation_level(generated_relations, 0, self.evaluation_level)
        labelled_relations = self.change_relation_level(labelled_relations, 0, self.evaluation_level)

        set_a = set(generated_relations)
        set_b = set(labelled_relations)
        # Find shared elements (Intersection)
        shared_relations = list(set_a & set_b)
        # Find elements only in generated_relations (Difference)
        only_generated_relations = list(set_a - set_b)
        # Find elements only in labelled_relations (Difference)
        only_labelled_relations = list(set_b - set_a)

        generated_relations_match = shared_relations + only_generated_relations + len(only_labelled_relations)*['stranger to x']
        labelled_relations_match = shared_relations + len(only_generated_relations)*['stranger to x'] + only_labelled_relations
        
        assert len(generated_relations_match) == len(labelled_relations_match)
        return generated_relations_match, labelled_relations_match, n_generated_ic, n_generated_ooc, n_labelled_ic, n_labelled_ooc

    def change_relation_level(self, relation_list, levelfrom, levelto):
        if levelto == 0:
            return relation_list
        new_list = []
        if (levelfrom == 0) and (levelto == 1):
            for relation in relation_list:
                if relation in self.equivalent_relation:
                    new_list.append(relation)
                else:
                    new_list.append(self.relation_merge_map[relation])
            return new_list
        elif (levelfrom == 1) and (levelto == 2):
            for relation in relation_list:
                if relation in self.high_level_relations:
                    new_list.append(relation)
                else:
                    new_list.append(self.high_level_category_map[relation])
            return new_list
        elif (levelfrom == 0) and (levelto == 2):
            return self.change_relation_level(self.change_relation_level(relation_list, 0, 1), 1, 2)
        else:
            raise Exception("Incorrect category mapping from %d to %d" % (levelfrom, levelto))

    def remove_corrupted_relations(self, relations):
        n_total = len(relations)
        relation_ic = []
        n_ic = 0
        for relation in relations:
            if relation in self.category_list:
                relation_ic.append(relation)
                n_ic += 1
        return relation_ic, n_ic, n_total-n_ic"""

    def cal_sim(self, generate_data_path, label_path, evaluate_type="merged", merged_file="all.json", results_to_save = {}):
        generated_similarity_scores = []
        for file in os.listdir(label_path):
            if file == ".DS_Store":
                continue
            else:
                for char_file in os.listdir(os.path.join(label_path, file)):
                    if file == ".DS_Store":
                        continue
                    else:
                        if char_file== "all.json":
                            generate_data_file_path = os.path.join(generate_data_path, file, merged_file)
                        else:
                            generate_data_file_path = os.path.join(generate_data_path, file, char_file)

                        if os.path.isfile(generate_data_file_path):
                            with open(generate_data_file_path, 'r') as f:
                                generated_data = json.load(f)
                                generated_data, _, _ = self.check_format(generated_data)
                                f.close()
                        else:
                            continue
                        with open(os.path.join(label_path, file, char_file), 'r') as f:
                            labelled_data = json.load(f)
                            labelled_data, _, _ = self.check_format(labelled_data)
                            f.close()

                    if evaluate_type == "merged":
                        if char_file == "all.json":
                            generated_similarity_scores = self.compare_sim(generated_data, labelled_data, generated_similarity_scores)
                    elif evaluate_type == "personal":
                        if file != "all.json":
                            generated_similarity_scores = self.compare_sim(generated_data, labelled_data, generated_similarity_scores)
                    else:
                        generated_similarity_scores = self.compare_sim(generated_data, labelled_data, generated_similarity_scores)
        if len(generated_similarity_scores) == 0:
            results_to_save[evaluate_type] = "na"
            return results_to_save
        else:
            results_to_save[evaluate_type] = sum(generated_similarity_scores) / len(generated_similarity_scores)
            return results_to_save

    def compare_sim(self, generated_data, labelled_data, generated_similarity_scores):
        for character, labelled_relations in labelled_data.items():
            labelled_relations = {item[0]: item[1] for item in labelled_relations}
            if character in generated_data:
                generated_relations ={item[0]: item[1] for item in generated_data[character]}
            else:
                generated_relations ={}
            for related_character in labelled_relations:
                if related_character in generated_relations:
                    generated_similarity_scores += self.cal_similarity_score(generated_relations[related_character], labelled_relations[related_character])
                else:
                    generated_similarity_scores += self.cal_similarity_score("stranger to X", labelled_relations[related_character])
        #print("check generated_similarity_scores", generated_similarity_scores)
        return generated_similarity_scores

    def cal_similarity_score(self, generated_relations, labelled_relations):
        generated_relations = [item.lower().strip().replace("-"," ") for item in generated_relations.split(",")] 
        labelled_relations = [item.lower().strip().replace("-"," ") for item in labelled_relations.split(",")] 
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        labelled_embeddings = model.encode(labelled_relations)
        generated_embedding = model.encode(generated_relations)

        output_score = []
        for relation in generated_embedding:
            #for compared_one in labelled_embeddings:
            #    embedding1_reshaped = relation.reshape(1, -1)
            #    embedding2_reshaped = compared_one.reshape(1, -1)
            relation_scores = [float(cosine_similarity(relation.reshape(1, -1), compared_one.reshape(1, -1))[0][0]) for compared_one in labelled_embeddings]
            output_score.append(max(relation_scores))
        #print("sim scores", generated_relations, labelled_relations, output_score)
        return output_score

    """def copy_folders_labelled(self):
        import shutil
        src_a = os.path.join(os.getcwd(), "data", "label")
        src_b = os.path.join(os.getcwd(), "data", "data_final_chinese")
        dest_c = os.path.join(os.getcwd(), "data", "data_labelled")

        if not os.path.exists(dest_c):
            os.makedirs(dest_c)

        for folder in os.listdir(src_a):
            if folder == ".DS_Store":
                continue

        for folder in os.listdir(src_a):
            path_a = os.path.join(src_a, folder)
            # Check if it's a directory
            if os.path.isdir(path_a):
                path_b = os.path.join(src_b, folder)
                # Check if the same folder exists in dir_b
                if os.path.exists(path_b) and os.path.isdir(path_b):
                    path_c = os.path.join(dest_c, folder)
                    if os.path.exists(path_c):
                        continue
                    else:
                        # Copy folder from dir_b to dir_c
                        shutil.copytree(path_b, path_c, dirs_exist_ok=True)

    def copy_folders_and_files(self, src_a, src_b, dest_c):
        import shutil
        # Ensure the destination directory exists
        if not os.path.exists(dest_c):
            os.makedirs(dest_c)

        # Copy folders from src_a to dest_c
        for folder in os.listdir(src_a):
            if folder == ".DS_Store":
                continue
            else:
                path = os.path.join(src_a, folder)
                if os.path.isdir(path):
                    shutil.copytree(path, os.path.join(dest_c, folder), dirs_exist_ok=True)

        for folder in os.listdir(src_b):
            src_path = os.path.join(src_b, folder)
            dest_path = os.path.join(dest_c, folder)
            if os.path.isdir(src_path) and os.path.exists(dest_path):
                for file in os.listdir(src_path):
                    src_file_path = os.path.join(src_path, file)
                    dest_file_path = os.path.join(dest_path, "all.json")
                    if os.path.isfile(src_file_path):
                        shutil.copy2(src_file_path, dest_file_path)"""

    """def get_all_stats(self):
        stats_to_save = {}
        # zero-shot baseline using 3.5, 4, llama-chat
        # keep update baseline using 3.5, 4, llama-chat
        # pairwise asking baseline using 3.5, 4, llama-chat
        # merge + identify confliction + retrieve evidence 3.5, 4, llama-chat
        # human annotation
        dir_torun = self.all_data_dir.copy()
        dir_torun['human'] = self.label_dir
        dir_torun['annotation'] = self.annotation_dir
        for check_dir in dir_torun:
            count_stats = self.get_label_stats(dir_torun[check_dir])
            # based on the chosen narrative in folder dir_torun[check_dir], get stats of the input narrative
            if check_dir != 'annotation':
                count_stats.update(self.get_document_stats(dir_torun[check_dir]))
            stats_to_save[check_dir] = count_stats

        stats_to_save = self.get_annotator_stats(stats_to_save)
        #stats_to_save["llm generated"] 
        with open(self.stats_path, 'w') as f:
            json.dump(stats_to_save, f, ensure_ascii=False, indent=4)
            f.close()"""
    
    def get_all_stats(self, language = "chinese"):
        stats_to_save = {}
        #label_dir = os.path.join(os.getcwd(), "data", language, "label")
        label_dir = os.path.join(os.getcwd(), "data", language, "gpt-4", "extract_whole_graph")
        count_stats = self.get_label_stats(label_dir)
        print(count_stats)
        doc_stats = self.get_document_stats(label_dir)
        print(doc_stats, doc_stats['narrative token']/doc_stats['narrative number'])
        self.get_story_length(label_dir)

    def count_character_number_all(self, language = "chinese"):
        model_list = ["gpt-4", "gpt-3.5-turbo", "llama-70b"]
        for model in model_list:
            check_dir = os.path.join(os.getcwd(), "data", language, model, "character")
            self.count_character_number(check_dir, os.path.join(os.getcwd(), "data", language, "label"))
    
    def count_character_number(self, check_dir, pick_dir):
        n_char = 0
        n_char_2 = 0
        for label_file in os.listdir(pick_dir):
            if label_file == ".DS_Store":
                continue
            else:
                for character_file in os.listdir(os.path.join(pick_dir, label_file)):
                    if label_file == ".DS_Store":
                        continue
                    else:
                        with open(os.path.join(check_dir, label_file, character_file), "r") as f:
                            all_character = json.load(f)
                            f.close()
                        if "characters" in all_character:
                            all_character = all_character["characters"]
                        n_char_2 += len(all_character)**2

                all_character_path = os.path.join(check_dir, label_file, "all.json")
                with open(all_character_path, "r") as f:
                    all_character = json.load(f)
                    f.close()
                if "characters" in all_character:
                    all_character = all_character["characters"]
                n_char += len(all_character)
                print(label_file, len(all_character), n_char)
        print(n_char, n_char/24, check_dir)
        print(n_char_2, n_char_2/24, check_dir)

    def check_format(self, data):
        checked_data = {}
        format_pass, format_error = 0, 0
        for  character, related_characters in data.items():
            checked_characters = []
            for related_character in related_characters:
                if not isinstance(related_character, list):
                    format_error += 1
                    continue
                if (len(related_character) == 2) and isinstance(related_character[1], str):
                    checked_characters.append(related_character)
                    format_pass += 1
                else:
                    format_error += 1
            if len(checked_characters) > 0:
                checked_data[character] = checked_characters
        return checked_data, format_pass, format_error
    
    def get_annotator_stats(self, stats_to_save):
        count_relation = {}
        count = stats_to_save["human"]["script_relation"]
        count.update(stats_to_save["annotation"]["script_relation"])
        annotator_path = os.path.join(os.getcwd(), "data", "annotator.json")
        with open(annotator_path, 'r') as f:
            script_annotators = json.load(f)
            f.close()
        for script, annotator in script_annotators.items():
            if annotator in count_relation:
                count_relation[annotator] += count[script]
            else:
                count_relation[annotator] = count[script]
        stats_to_save["annotator count"] = count_relation
        total = 0
        count_relation["hn"] += 303
        print(count_relation)
        for key, count in count_relation.items():
            total += count
        for key, count in count_relation.items():
            print(key, count*54/total, total)
        return stats_to_save

    def get_label_stats(self, check_dir):
        """
        For all script we saved, extract the relationships and merge the graph
        """
        
        all_relations = {}
        all_narrative = {}
        n_annotation_narrative = {}
        format_pass_total, format_error_total, out_of_input_limit = 0, 0, 0
        count_stats = {"empty file": 0, "total file": 0}
        for label_file in os.listdir(check_dir):
            if label_file == ".DS_Store":
                continue
            #elif label_file == "all.json":
            else:
                character_list = []
                character_with_narrative = []
                for character_file in os.listdir(os.path.join(check_dir, label_file)):
                    if not os.path.isfile(os.path.join(check_dir, label_file, "all.json")):
                        out_of_input_limit += 1
                    if label_file == ".DS_Store":
                        continue
                    else:
                        if (label_file != "all.json") and (label_file != "update_all.json"):
                            character_with_narrative.append(character_file[:-4])
                    print(os.path.join(check_dir, label_file, character_file))
                    with open(os.path.join(check_dir, label_file, character_file), "r") as f:
                        data = json.load(f)
                        f.close()
                    if not data:
                        count_stats["empty file"] += 1
                    else:
                        count_stats["total file"] += 1
                    print(label_file, character_file)
                    data, format_pass, format_error = self.check_format(data)
                    format_pass_total += format_pass
                    format_error_total += format_error
                    character_list += list(data.keys())
                    for character in data:
                        if character == "characters":
                            continue
                        else:
                            relation_list = data[character]
                            for relations in relation_list:
                                for relation in relations[1].split(","):
                                    relation = relation.strip().lower().replace("-"," ")
                                    if label_file in n_annotation_narrative:
                                        n_annotation_narrative[label_file] += 1
                                    else:
                                        n_annotation_narrative[label_file] = 1
                                    if relation in all_relations:
                                        all_relations[relation] += 1
                                    else:
                                        all_relations[relation] = 1
                                        if relation not in self.category_list:
                                            print("OOD!!!!", relation, label_file)
                all_narrative[label_file] = [len(list(set(character_list))), len(character_with_narrative)]

        all_relations = {k: v for k, v in sorted(all_relations.items(), key=lambda item: item[1])}
        count = {"ic type": 0, "ic number": 0, "ooc type": 0, "ooc number": 0}
        for relation in all_relations:
            if relation in self.category_list:
                #print(relation, all_relations[relation])
                count["ic type"] += 1
                count["ic number"] += all_relations[relation]
            else:
                #print("OOD!!!!", relation, all_relations[relation])
                count["ooc type"] += 1
                count["ooc number"] += all_relations[relation]
        count["ooc rate"] = count["ooc number"]/(count["ooc number"] + count["ic type"])
        count["ooil"] = out_of_input_limit
        count["total"] = count["ic number"] + count["ooc number"]
        count["script_relation"] = n_annotation_narrative

        num_char = {}
        # Calculate the average #characters per narrative
        sum_of_first_elements = sum(all_narrative[narrative][0] for narrative in all_narrative)
        num_char["average all"] = sum_of_first_elements / len(list(all_narrative.keys()))
        # Calculate the average #characters with narrative per narrative
        sum_of_first_elements = sum(all_narrative[narrative][1] for narrative in all_narrative)
        num_char["average with narrative"] = sum_of_first_elements / len(list(all_narrative.keys()))
        num_char["average without narrative"] = num_char["average all"] - num_char["average with narrative"]

        # save stats
        count_stats.update(count)
        count_stats.update(num_char)
        format_stats = {"correctly formatted relation": format_pass_total, "incorrectly formatted relation": format_error_total, "pass rate": format_pass_total/(format_pass_total+format_error_total)}
        count_stats.update(format_stats)
        #count_stats["all_relations"] = all_relations
        #count_stats["all_narrative"] = all_narrative
        
        return count_stats
        
    def get_document_stats(self, check_dir):
        doc_stats = {"narrative token": 0, "narrative number": 0}
        for label_file in os.listdir(check_dir):
            if label_file == ".DS_Store":
                continue
            for character_file in os.listdir(os.path.join(self.input_narrative, label_file, "txt")):
                if character_file == ".DS_Store":
                    continue
                else:
                    #print(os.path.join(self.input_narrative, label_file, "txt", character_file))
                    with open(os.path.join(self.input_narrative, label_file, "txt", character_file)) as f:
                        character_background_text = f.readlines()
                        f.close()
                    character_background_text = " ".join(character_background_text)
                    doc_stats["narrative token"] += count_string_tokens(character_background_text)
                    doc_stats["narrative number"] += 1
        return doc_stats
    
    def get_story_length(self, check_dir):
        import csv
        story_length = {}
        individual_length = {}
        for label_file in os.listdir(check_dir):
            if label_file == ".DS_Store":
                continue
            else:
                story_length[label_file] = 0
            for character_file in os.listdir(os.path.join(self.input_narrative, label_file, "txt")):
                if character_file == ".DS_Store":
                    continue
                else:
                    #print(os.path.join(self.input_narrative, label_file, "txt", character_file))
                    with open(os.path.join(self.input_narrative, label_file, "txt", character_file)) as f:
                        character_background_text = f.readlines()
                        f.close()
                    character_background_text = " ".join(character_background_text)
                    individual_length[label_file+"_"+character_file] = count_string_tokens(character_background_text)
                    story_length[label_file] += count_string_tokens(character_background_text)
        story_length = dict(sorted(story_length.items()))
        individual_length = dict(sorted(individual_length.items()))
        print(story_length)

        csv_file_path = os.path.join(os.getcwd(), "data", "evaluate", 'story_length.csv')
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Key', 'Value'])
            for key, value in story_length.items():
                writer.writerow([key, value])

        csv_file_path = os.path.join(os.getcwd(), "data", "evaluate", 'individual_length.csv')
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Key', 'Value'])
            for key, value in individual_length.items():
                writer.writerow([key, value])

        return story_length