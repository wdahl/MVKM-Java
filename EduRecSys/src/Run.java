import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Set;

import org.json.simple.JSONObject;
import org.json.simple.JSONArray;

import org.apache.commons.math3.stat.correlation.SpearmansCorrelation;

import java.io.File;  // Import the File class
import java.io.FileWriter;
import java.io.IOException;  // Import the IOException class to handle errors


public class Run{
    public static void sequentialRecommendation(String dataStr, String courseStr, String modelStr,
                                    Integer fold, Integer skillDim, Integer conceptDim,
                                    Double lambda_s, Double lambda_t, Double lambda_q,
                                    Double lambda_l, Double lambda_d, Double lambda_bias,
                                    Double penalty_weight, Integer markovian, Double lectureTradeOff,
                                    Double discussionTradeOff, Double lr, Integer max_iter,
                                    Integer top_k, Integer startAttempt, Integer endAttempt,
                                    String[] metrics, String logFile, String remark) throws Exception{
        /**
         * pipeline of running single experiment with 5-fold cross validation
         * @para: a list of parameters for a single case of experiment
         * @return: void
         */
        
        String views;
        if(courseStr.equals("Quiz")){
            views = "100";
        }
        else if(courseStr.equals("Quiz_Lecture")){
            views = "110";
        }
        else if(courseStr.equals("Quiz_Discussion")){
            views = "101";
        }
        else if(courseStr.equals("Quiz_Lecture_Discussion")){
            views = "111";
        }
        else{
            throw new Exception("IOError");
        }

        DataHelper config = new DataHelper(dataStr, courseStr, fold, views, skillDim, conceptDim, lambda_s, lambda_t, lambda_q, lambda_l, lambda_d, lambda_bias, penalty_weight, markovian, lectureTradeOff, discussionTradeOff, lr, max_iter, top_k, startAttempt, endAttempt, metrics, logFile);
        ArrayList<ArrayList<Number>> testData = config.testSet;
        System.out.println("========================================================");
        
        ProximalTutor tutor = new ProximalTutor(config);
        
        tutor.training();
        for(Integer testAttempt=startAttempt; testAttempt<endAttempt; testAttempt++){
        	tutor.startAttempt = testAttempt;
            tutor.lr = lr;
            // restart_training(model)
            ArrayList<Double> train_perf = tutor.fastTraining();

            ArrayList<ArrayList<Number>> test_set = new ArrayList<>();
            for(ArrayList<Number> innerList : testData){
            	ArrayList<Number> l = new ArrayList<>();
                Integer student = (Integer) innerList.get(0);
                l.add(student);
                Integer attempt = (Integer) innerList.get(1);
                l.add(attempt);
                Integer material = (Integer) innerList.get(2);
                l.add(material);
                Double score = (Double) innerList.get(3);
                l.add(score);
                Integer resource = (Integer) innerList.get(4);
                l.add(resource);

                if(attempt == tutor.startAttempt){
                	test_set.add(l);
                    tutor.trainSet.add(l);
                }
            }
            tutor.testing(test_set);
            tutor.generateNextItems();
            System.out.println(testAttempt + " " + endAttempt);
        }
        
        LinkedHashMap<String, Double> overall_perf = tutor.eval(tutor.testObsList, tutor.testPredList);
        JSONObject perf_dict = mapToJSON(overall_perf);
  
        Double correlation = offlineEvaluation(tutor);
        
        String result_file_path = String.format("results/%s/%s/%s/eval_results.csv", dataStr, courseStr, modelStr);

        try {
            File file = new File(result_file_path);
            if (file.createNewFile()) {
                FileWriter myWriter = new FileWriter(result_file_path);
                myWriter.write("fold, top_k, rmse, spearman correlation\n");
                myWriter.close();
            }
        }
        catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }

        try {
            FileWriter myWriter = new FileWriter(result_file_path, true);
            myWriter.write(fold + "," + top_k + "," + overall_perf.get("rmse") + "," + correlation + "\n");
            myWriter.close();
        } 
        catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
        
        perf_dict.put("hit_count", tutor.total_hit);
        perf_dict.put("miss_count", tutor.total_miss);
        perf_dict.put("correlation", correlation);
        saveExpResults(tutor, perf_dict, dataStr, courseStr, modelStr,
        		fold, skillDim, conceptDim, lambda_s, lambda_t,
                lambda_q, lambda_bias, penalty_weight, markovian,
                lr, max_iter, top_k, startAttempt,
                endAttempt);
    }
    
    public static Double offlineEvaluation(ProximalTutor tutor) {
    	ArrayList<Double> avg_ipw_list = new ArrayList<>();
    	ArrayList<Double> sip_list = new ArrayList<>();
    	
    	ArrayList<Integer> sortedUsers = new ArrayList<>();
    	sortedUsers.addAll(tutor.recommendation.keySet());
    	Collections.sort(sortedUsers);
    	
    	for(Integer user : sortedUsers) {
    		System.out.println("------------------------------------------------------------------------------------");
    		Set<Integer> question_list = new HashSet<>();
    		for(ArrayList<Number> innerList : tutor.userData.get(user)) {
    			Integer time_index = (Integer) innerList.get(1);
    			Integer question = (Integer) innerList.get(2);
    			Double score = (Double) innerList.get(3);
    			
    			question_list.add(question);
    		}
    		
    		System.out.println(String.format("student %s, number of attempts: %d, number of questions: %d", user.toString(), tutor.userData.get(user).size(), question_list.size()));
    		
    		Double avg_ipw = mean(tutor.recommendation.get(user));
    		avg_ipw_list.add(avg_ipw);
    		
    		Double sip = 0.0;
    		System.out.print("histroical records: {");
    		for(Integer question : tutor.historicalRecords.get(user).keySet()) {
    			System.out.print(question + ": ");
    			ArrayList<Double> temp = tutor.historicalRecords.get(user).get(question);
    			System.out.print(temp + ", ");
    			Double final_question_score = temp.get(temp.size()-1);
    			Integer num_trail = temp.size();
    			sip += final_question_score / log2(num_trail + 1.0);
    		}
    		System.out.println("}");
    		
    		sip_list.add(sip);
    		
    		System.out.println(String.format("avg. ipw-dch %s and students's SIP score : %s", avg_ipw.toString(), sip.toString()));
    		System.out.println("--------------------------------------------------------------------------------------------------------------");
    	}
    	
    	System.out.println("distrobution of recommended items:");
    	System.out.println("question: count");
    	
    	for(Integer question : tutor.recMap.keySet()) {
    		Integer count = tutor.recMap.get(question);
    		System.out.println(question + ": " + count);
    	}
    	
    	System.out.println("==============================================================================================================================");
    	
    	SpearmansCorrelation spear = new SpearmansCorrelation();
    	Double correlation = spear.correlation(listToArray(avg_ipw_list), listToArray(sip_list));
    	
    	System.out.println(String.format("user's avg. IPW_DCH list vs proposed SIP metric sperarman correlation %s", correlation.toString()));
    	
    	return correlation;
    }

    public static void saveExpResults(ProximalTutor model, JSONObject perfMap, String dataStr,
                                String courseStr, String modelStr, Integer fold, Integer skill_dim,
                                Integer concept_dim, Double lambda_s, Double lambda_t, Double lambda_q,
                                Double lambda_bias, Double penalty_weight, Integer markovian,
                                Double lr, Integer max_iter, Integer top_k, Integer startAttempt, Integer endAttempt){

        String result_file_path = String.format("results/%s/%s/%s/fold_%s_test_results.json", dataStr, courseStr, modelStr, fold.toString());

        try {
            File file = new File(result_file_path);
            if(file.createNewFile()){
                FileWriter writer = new FileWriter(file);

                JSONObject result = new JSONObject();
                result.put("skill_dim", skill_dim);
                result.put("concept_dim", concept_dim);
                result.put("lambda_s", lambda_s);
                result.put("lambda_t", lambda_t);
                result.put("lambda_q", lambda_q);
                result.put("lambda_bias", lambda_bias);
                result.put("penalty_weight", penalty_weight);
                result.put("markovian_steps", markovian);
                result.put("learning_rate", lr);
                result.put("max_iter", max_iter);
                result.put("top_k", top_k);
                result.put("start_attempt", startAttempt);
                result.put("end_attempt", endAttempt);
                result.put("perf", perfMap);

                writer.write(result.toJSONString());
                writer.close();
            }
        }
        catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }

    public static void run_morf(String modelStr) throws Exception{
        String dataStr = "morf";
        String courseStr = "Quiz";
        String[] metrics = {"rmse", "mae"};

        String remark = null;
        Integer skillDim = 3;
        Integer conceptDim = 5;
        Double lambda_s = 0.0;
        Double lambda_t = 0.1;
        Double lambda_q = 0.0;
        Double lambda_l = 0.0;
        Double lambda_d = 0.0;
        Double lambda_bias = 0.0;
        Double penalty_weight = 0.1;
        Integer markovian = 1;
        Double lectureTradeOff = 0.0;
        Double discussionTradeOff = 0.0;
        Double lr = 0.1;
        Integer max_iter = 10;
        Integer startAttempt = 1;
        Integer endAttempt = 25;

        Integer num_folds = 1;

        Integer[] top_k_list = {3};
        for(Integer fold=1; fold<=num_folds; fold++){
            for(Integer top_k : top_k_list){
                sequentialRecommendation(dataStr, courseStr, modelStr, fold, skillDim, conceptDim, lambda_s, lambda_t, lambda_q, lambda_l, lambda_d, lambda_bias, penalty_weight, markovian, lectureTradeOff, discussionTradeOff, lr, max_iter, top_k, startAttempt, endAttempt, metrics, null, remark);
            }
        }
    }
    
    public static void run_mastery_grids(String modelStr) throws Exception {
    	String dataStr = "mastery_grids";
    	String courseStr = "Quiz";
    	String[] metrics = {"rmse", "mae"};
    	
    	String remark = null;
        Integer skillDim = 3;
        Integer conceptDim = 5;
        Double lambda_s = 0.0;
        Double lambda_t = 0.0;
        Double lambda_q = 0.0;
        Double lambda_l = 0.0;
        Double lambda_d = 0.0;
        Double lambda_bias = 0.0;
        Double penalty_weight = 0.1;
        Integer markovian = 1;
        Double lectureTradeOff = 0.0;
        Double discussionTradeOff = 0.0;
        Double lr = 0.1;
        Integer max_iter = 10;
        Integer startAttempt = 1;
        Integer endAttempt = 70;

        Integer num_folds = 1;
        
        Integer[] top_k_list = {3};
        for(Integer fold=1; fold<=num_folds; fold++){
            for(Integer top_k : top_k_list){
                sequentialRecommendation(dataStr, courseStr, modelStr, fold, skillDim, conceptDim, lambda_s, lambda_t, lambda_q, lambda_l, lambda_d, lambda_bias, penalty_weight, markovian, lectureTradeOff, discussionTradeOff, lr, max_iter, top_k, startAttempt, endAttempt, metrics, null, remark);
            }
        }
    }

    public static void main(String[] args) throws Exception{
        String modelStr = "proximal";
        run_morf(modelStr);
        //run_mastery_grids(modelStr);
    }
    
    private static JSONObject mapToJSON(LinkedHashMap<String, Double> map){
        JSONObject json = new JSONObject();
        for(String key : map.keySet()){
            json.put(key, map.get(key));
        }

        return json;
    }

    private static JSONArray listToJSON(ArrayList<Double> list){
        JSONArray json = new JSONArray();
        for(Double d : list){
            json.add(d);
        }

        return json;
    }

    private static Double mean(List<Double> l){
        Double sum = 0.0;
        for(Double d : l){
            sum += d;
        }

        return sum/l.size();
    }

    private static Double[] meanAxis0(ArrayList<ArrayList<Number>> data){
        Double studentSum = 0.0;
        Double attemptSum = 0.0;
        Double questionSum = 0.0;
        Double obsSum = 0.0;
        Double resourceSum = 0.0;
        for(ArrayList<Number> innerList : data){
            Integer student = (Integer) innerList.get(0);
            Integer attempt = (Integer) innerList.get(1);
            Integer question = (Integer) innerList.get(2);
            Double obs = (Double) innerList.get(3);
            Integer resource = (Integer) innerList.get(4);

            studentSum += Double.valueOf(student);
            attemptSum += Double.valueOf(attempt);
            questionSum += Double.valueOf(question);
            obsSum += obs;
            resourceSum += Double.valueOf(resource);
        }

        Double size = Double.valueOf(data.size());
        Double[] means = {studentSum/size, attemptSum/size, questionSum/size, obsSum/size, resourceSum/size};
        return means;
    }

    private static Double log2(Double x){
        return Math.log(x) / Math.log(2);
    }

    private static double[] listToArray(ArrayList<Double> list){
        double[] arr = new double[list.size()];
        for(int i=0; i<list.size(); i++){
            arr[i] = list.get(i);
        }

        return arr;
    }

    private static Double[][] randomSample(Integer n, Integer m){
        Double[][] matrix = new Double[n][m];
        for(int i=0; i<n; i++){
            for(int j=0; j<m; j++){
                matrix[i][j] = Math.random();
            }
        }

        return matrix;
    }

    private static Double[][][] randomSample(Integer n, Integer m, Integer d){
        Double[][][] tensor = new Double[n][m][d];
        for(int i=0; i<n; i++){
            for(int j=0; j<m; j++){
                for(int k=0; j<d; k++){
                    tensor[i][j][k] = Math.random();
                }
            }
        }

        return tensor;
    }

    private static Double[][] zeros(Integer n, Integer m){
        Double[][] matrix = new Double[n][m];
        for(int i=0; i<n; i++){
            for(int j=0; j<m; j++){
                matrix[i][j] = 0.0;
            }
        }

        return matrix;
    }

    private static Double[] zeros(Integer n){
        Double[] vector = new Double[n];
        for(int i=0; i<n; i++){
            vector[i] = 0.0;
        }

        return vector;
    }
}