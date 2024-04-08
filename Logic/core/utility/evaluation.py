import numpy
from typing import List
import wandb

class Evaluation:

    def __init__(self, name: str):
            self.name = name

    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The precision of the predicted results
        """

        true_pos = 0
        total_pos = 0

        for i in range(len(actual)):
            actual_set = set(actual[i])
            predicted_set = set(predicted[i])
            true_positives = len(actual_set.intersection(predicted_set))
            
            true_pos += true_positives
            total_pos += len(predicted_set)


        return true_pos / total_pos

        

    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the recall of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The recall of the predicted results
        """

        
        true_pos = 0
        total_pos = 0

        for i in range(len(actual)):
            actual_set = set(actual[i])
            predicted_set = set(predicted[i])
            true_positives = len(actual_set.intersection(predicted_set))
        
            true_pos += true_positives
            total_pos += len(actual_set)


        return true_pos / total_pos
    
    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the F1 score of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The F1 score of the predicted results    
        """
        p = self.calculate_precision(actual, predicted)
        r = self.calculate_recall(actual, predicted)
        f1 = 2 * p * r / (p+r)

        return f1
    
    def calculate_AP(self, actual: List[str], predicted: List[str]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Average Precision of the predicted results
        """
        AP = 0.0

        true_pos = 0

        for i in range(len(predicted)):
            if predicted[i] in actual:
                true_pos += 1
                AP += true_pos/(i+1)
            


        return AP/true_pos
    
    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Mean Average Precision of the predicted results
        """
        MAP = 0.0

        for i in range(len(actual)):
            MAP += self.calculate_AP(actual[i],predicted[i])

        return MAP / len(actual)
    
    def cacluate_DCG(self, actual: List[str], predicted: List[str]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The DCG of the predicted results
        """
        DCG = 0

        for i in range(len(predicted)):
            if predicted[i] in actual:
                if i==0:
                    DCG += len(actual) - actual.index(predicted[i])
                else:
                    DCG += (len(actual) - actual.index(predicted[i]))/numpy.log2(i+1)
        
        return DCG 
    
    def cacluate_NDCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The NDCG of the predicted results
        """
        NDCG = 0.0

        for i in range(len(actual)):
            NDCG += self.cacluate_DCG(actual[i],predicted[i]) / self.cacluate_DCG(actual[i],actual[i])

        return NDCG / len(actual)
    
    def cacluate_RR(self, actual: List[str], predicted: List[str]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Reciprocal Rank of the predicted results
        """
        

        for i in range(len(predicted)):
            if predicted[i] in actual:
                return 1/(i+1)

        return 0
    
    def cacluate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The MRR of the predicted results
        """
        MRR = 0.0

        for i in range(len(actual)):
            MRR += self.cacluate_RR(actual[i], predicted[i])

        return MRR/len(actual)
    

    def print_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Prints the evaluation metrics

        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        print(f"name = {self.name}")

        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")
        
        print(f"MAP: {map}")
        
        print(f"NDCG: {ndcg}")
        
        print(f"MRR: {mrr}")
      

    def log_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Use Wandb to log the evaluation metrics
      
        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        
        wandb.init(project="mir-project")

        wandb.log({
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            # "Average Precision": ap,
            "Mean Average Precision": map,
            # "Discounted Cumulative Gain": dcg,
            "Normalized Discounted Cumulative Gain": ndcg,
            # "Reciprocal Rank": rr,
            "Mean Reciprocal Rank": mrr
        })


    def calculate_evaluation(self, actual: List[List[str]], predicted: List[List[str]]):
        """
        call all functions to calculate evaluation metrics

        parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results
            
        """

        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = self.calculate_F1(actual, predicted)
        # ap = self.calculate_AP(actual, predicted)
        map_score = self.calculate_MAP(actual, predicted)
        # dcg = self.cacluate_DCG(actual, predicted)
        ndcg = self.cacluate_NDCG(actual, predicted)
        # rr = self.cacluate_RR(actual, predicted)
        mrr = self.cacluate_MRR(actual, predicted)
        ap = 0
        dcg = 0
        rr = 0

        #call print and viualize functions
        self.print_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)
        self.log_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)



eval = Evaluation('test')
eval.calculate_evaluation([['batman','the batman']],[['dark knight','batman']])