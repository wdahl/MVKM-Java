import org.json.simple.JSONObject;
import org.json.simple.JSONArray;
import org.json.simple.parser.ParseException;
import org.json.simple.parser.JSONParser;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.BufferedReader;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.LinkedHashMap; 
import java.util.Map; 
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.Arrays;;

public class DataHelper{
	String dataStr;
    String views;
    Integer numUsers;
    Integer numAttempts;
    Integer numQuestions;
    Integer numLectures;
    Integer numDiscussions;
    Integer numSkills;
    Integer numConcepts;
    Double lambda_s;
    Double lambda_t;
    Double lambda_q;
    Double lambda_l;
    Double lambda_d;
    Double lambda_bias;
    Double penality_weight;
    Integer markovian;
    Double lectureTradeOff;
    Double discussionTradeOff;
    Double lr;
    Integer iter;
    Double tol;
    Integer top_k;
    Integer startAttempt;
    Integer endAttempt;
    String[] metrics;
    String logfile;
    LinkedHashMap<Integer, ArrayList<ArrayList<Number>>> userData;
    ArrayList<ArrayList<Number>> trainSet;
    ArrayList<ArrayList<Number>> testSet;
    LinkedHashMap<Integer, Boolean> testUsers;
    LinkedHashMap<Integer, LinkedHashMap<Integer, Integer>> testUserRecords;
    LinkedHashMap<Integer, ArrayList<Double>>testUserTrainingPeriod;
    LinkedHashMap<Integer, ArrayList<Double>> testUserTestingPeriod;
    LinkedHashMap<Integer, ArrayList<Double>> questionScoresMap;
    LinkedHashMap<Integer, LinkedHashMap<Integer, Integer>> nextQuestionsMap;

    /**
     *  generate model configurations for training and testing
        such as initialization of each parameters and hyperparameters
        :return: config dict
     * @throws Exception 
     */
    public DataHelper(String dataStr, String courseStr, Integer fold, String views, Integer skill,
                    Integer concept, Double lambda_s, Double lambda_t, Double lambda_q, 
                    Double lambda_l, Double lambda_d, Double lambda_bias, Double penality_weight,
                    Integer markovian, Double lectureTradeOff, Double discussionTradeOff, Double lr, Integer iter, 
                    Integer top_k, Integer startAttempt, Integer endAttempt, String[] metrics, String logfile) throws Exception
    {
        if(top_k == null){
            top_k = 3;
        }

        if(startAttempt == null){
            startAttempt = 1;
        }

        //JSON parser object to parse read file
        JSONParser jsonParser = new JSONParser();
        JSONObject data = null;
        
        // Trys reading the data from a json file
        try (FileReader reader = new FileReader(String.format("data/%s/%s/%d_train_val_test.json", dataStr, courseStr, fold)))
        {
            //Read JSON file
            data = (JSONObject) jsonParser.parse(reader);
            reader.close();
        } 
        catch (FileNotFoundException e) 
        {
            e.printStackTrace();
        } 
        catch (IOException e) 
        {
            e.printStackTrace();
        } 
        catch (ParseException e) 
        {
            e.printStackTrace();
        }
        
        Integer numAttempts = ((Long)data.get("num_attempts")).intValue();
        System.out.println("Number of attempts: " + numAttempts.toString());

        if(endAttempt == null){
            endAttempt = numAttempts;
        }
        else if(endAttempt < numAttempts){
            numAttempts = endAttempt;
        }
        else{
            endAttempt = numAttempts;
        }

        // generate config, train_set, test_set for general train and test
        LinkedHashMap<Integer, ArrayList<ArrayList<Number>>> userData = new LinkedHashMap<>();
        ArrayList<ArrayList<Number>> testData = new ArrayList<>();
        LinkedHashMap<Integer, Boolean> testUsers = new LinkedHashMap<>();

        // Creates Testing data set
        JSONArray dataTest = (JSONArray) data.get("test");
        for(int i=0; i<dataTest.size(); i++){
            JSONArray innerJSONArray = (JSONArray) dataTest.get(i);
            ArrayList<Number> innerArrayList = new ArrayList<>();

            Integer student = ((Long)innerJSONArray.get(0)).intValue();
            innerArrayList.add(student);
            Integer attempt = ((Long)innerJSONArray.get(1)).intValue();
            innerArrayList.add(attempt);
            Integer question = ((Long)innerJSONArray.get(2)).intValue();
            innerArrayList.add(question);
            Double score = (double) innerJSONArray.get(3);
            innerArrayList.add(score);
            Integer resource = ((Long)innerJSONArray.get(4)).intValue();
            innerArrayList.add(resource);
            
            if(attempt < endAttempt) {
            	testData.add(innerArrayList);
            	if(!userData.containsKey(student)){
                    userData.put(student, new ArrayList<>());
                }
            	
            	userData.get(student).add(innerArrayList);
            	if(!testUsers.containsKey(student)){
                    testUsers.put(student, true);
                }
            }
        }

        // Creates training data set
        ArrayList<ArrayList<Number>> trainData = new ArrayList<>();
        LinkedHashMap<Integer, Boolean> trainUsers = new LinkedHashMap<>();
        JSONArray dataTrain = (JSONArray) data.get("train");
        for(int i=0; i<dataTrain.size(); i++){
            JSONArray innerJSONArray = (JSONArray) dataTrain.get(i);
            ArrayList<Number> innerArrayList = new ArrayList<>();

            Integer student = ((Long)innerJSONArray.get(0)).intValue();
            innerArrayList.add(student);
            Integer attempt = ((Long)innerJSONArray.get(1)).intValue();
            innerArrayList.add(attempt);
            Integer question = ((Long)innerJSONArray.get(2)).intValue();
            innerArrayList.add(question);
            Double score = (double) innerJSONArray.get(3);
            innerArrayList.add(score);
            Integer resource = ((Long)innerJSONArray.get(4)).intValue();
            innerArrayList.add(resource);

            if(attempt < endAttempt) {
            	trainData.add(innerArrayList);
            	if(!userData.containsKey(student)){
                    userData.put(student, new ArrayList<>());
                }
            	userData.get(student).add(innerArrayList);

                if(!trainUsers.containsKey(student) && !testUsers.containsKey(student)){
                    trainUsers.put(student, true);
                }
            }
        }

        // Gets value data set from the training
        JSONArray dataVal = (JSONArray) data.get("val");
        for(int i=0; i<dataVal.size(); i++){
            JSONArray innerJSONArray = (JSONArray) dataVal.get(i);
            ArrayList<Number> innerArrayList = new ArrayList<>();

            Integer student = ((Long)innerJSONArray.get(0)).intValue();
            innerArrayList.add(student);
            Integer attempt = ((Long)innerJSONArray.get(1)).intValue();
            innerArrayList.add(attempt);
            Integer question = ((Long)innerJSONArray.get(2)).intValue();
            innerArrayList.add(question);
            Double score = (double) innerJSONArray.get(3);
            innerArrayList.add(score);
            Integer resource = ((Long)innerJSONArray.get(4)).intValue();
            innerArrayList.add(resource);

            
            
            if(attempt < endAttempt) {
            	trainData.add(innerArrayList);
            	if(!userData.containsKey(student)){
                    userData.put(student, new ArrayList<>());
                }
                userData.get(student).add(innerArrayList);

                if(!trainUsers.containsKey(student) && !testUsers.containsKey(student)){
                    trainUsers.put(student, true);
                }
            }
        }

        LinkedHashMap<Integer, LinkedHashMap<Integer, Integer>> nextQuestionsMap = new LinkedHashMap<>();
        for(Integer user : userData.keySet()){
        	ArrayList<ArrayList<Number>> records = userData.get(user);
            records.sort(new Comparator<ArrayList<Number>>(){
                    public int compare(ArrayList<Number> l1,  ArrayList<Number> l2){
                    	Integer i = (Integer) l1.get(1);
                    	Integer j = (Integer) l2.get(1);
                        return i.compareTo(j);
                    }
                }
            );

            userData.replace(user, records);
            for(int index=0; index<records.size()-1; index++){
                Integer student = (Integer) records.get(index).get(0);
                Integer attempt = (Integer) records.get(index).get(1);
                Integer question = (Integer) records.get(index).get(2);
                Double score = (Double) records.get(index).get(3);
                Integer resource = (Integer) records.get(index).get(4);

                if(!nextQuestionsMap.containsKey(question)){
                    nextQuestionsMap.put(question, new LinkedHashMap<>());
                }

                Integer nextQuestion = (Integer) records.get(index+1).get(2);
                if(!nextQuestionsMap.get(question).containsKey(nextQuestion)){
                    nextQuestionsMap.get(question).put(nextQuestion, 0);
                }

                nextQuestionsMap.get(question).replace(nextQuestion, nextQuestionsMap.get(question).get(nextQuestion)+1);

            }
        }

        ArrayList<ArrayList<Number>> trainSet = new ArrayList<>();
        ArrayList<ArrayList<Number>> testSet = new ArrayList<>();
        LinkedHashMap<Integer, LinkedHashMap<Integer, Integer>> testUserRecords = new LinkedHashMap<>();

        LinkedHashMap<Integer, ArrayList<Double>>testUserTrainingPeriod = new LinkedHashMap<>();
        LinkedHashMap<Integer, ArrayList<Double>> testUserTestingPeriod = new LinkedHashMap<>();

        LinkedHashMap<Integer, ArrayList<Double>> questionScoresMap = new LinkedHashMap<>();
        LinkedHashMap<Integer, LinkedHashMap<Integer, ArrayList<Double>>> userRecordsByQuestions = new LinkedHashMap<>();

        ArrayList<Integer> userDataArray = new ArrayList<>();
        userDataArray.addAll(userData.keySet());
        Collections.sort(userDataArray);
        for(Integer user : userDataArray){
        	ArrayList<ArrayList<Number>> records = new ArrayList<>();
        	if(userData.get(user).size() < endAttempt) {
        		records.addAll(userData.get(user));
        	}
        	else {
        		records.addAll(userData.get(user).subList(0, endAttempt));
        	}
        	
            if(!userRecordsByQuestions.containsKey(user)){
                userRecordsByQuestions.put(user, new LinkedHashMap<>());
            }

            // Build questionScoresMap
            for(ArrayList<Number> record : records){
                Integer attempt = (Integer) record.get(1);
                Integer question = (Integer) record.get(2);
                Double outcome = (Double) record.get(3);
                Integer resource = (Integer) record.get(4);

                if(!questionScoresMap.containsKey(question)){
                    questionScoresMap.put(question, new ArrayList<>());
                }

                questionScoresMap.get(question).add(outcome);

                if(!userRecordsByQuestions.get(user).containsKey(question)){
                    userRecordsByQuestions.get(user).put(question, new ArrayList<>());
                }
                
                userRecordsByQuestions.get(user).get(question).add(outcome);
            }

            System.out.print("User: ");
            System.out.println(user);
            for(Integer question : userRecordsByQuestions.get(user).keySet()){
                ArrayList<Double> scores = userRecordsByQuestions.get(user).get(question);
                if(isIncreasing(scores)){
                    System.out.print("PASS! ");
                }
                else{
                    System.out.print("FAIL! ");
                }
                System.out.println("question: " + question.toString() + " scores: " + scores.toString());
            }

            if(testUsers.containsKey(user)){
                trainSet.addAll(records.subList(0, startAttempt));
                testSet.addAll(records.subList(startAttempt, records.size()));

                // generate test user records dictionary for quick access
                if(!testUserRecords.containsKey(user)){
                    testUserRecords.put(user, new LinkedHashMap<>());
                }

                if(!testUserTrainingPeriod.containsKey(user)){
                    testUserTrainingPeriod.put(user, new ArrayList<>());
                }

                if(!testUserTestingPeriod.containsKey(user)){
                    testUserTestingPeriod.put(user, new ArrayList<>());
                }

                for(ArrayList<Number> record : records){
                    Integer attempt = (Integer) record.get(1);
                    Integer question = (Integer) record.get(2);
                    Double outcome = (Double) record.get(3);
                    Integer resource = (Integer) record.get(4);

                    testUserRecords.get(user).put(attempt, question);
                    if(attempt < startAttempt){
                        testUserTrainingPeriod.get(user).add(outcome);
                    }
                    else{
                        testUserTestingPeriod.get(user).add(outcome);
                    }
                }
            }
            else{
                trainSet.addAll(records);
            }
        }
        
        System.out.println("=======================================================");
        System.out.println("train data: " + trainData.size());
        System.out.println("test data: " + testData.size());
        System.out.println("number of train users: " + trainUsers.size());
        System.out.println("number of test users: " + testUsers.size());
        
        System.out.println("Question: Mean Score, STD");
        for(Integer question : questionScoresMap.keySet()){
            System.out.println(question + " " + mean(questionScoresMap.get(question)) + " " + std(questionScoresMap.get(question)));
        }

        this.dataStr = dataStr;
        this.views = views;
        this.numUsers = ((Long)data.get("num_users")).intValue();
        this.numAttempts = numAttempts;
        this.numQuestions = ((Long)data.get("num_quizzes")).intValue();
        this.numLectures = ((Long)data.get("num_lectures")).intValue();
        this.numDiscussions = ((Long)data.get("num_discussions")).intValue();
        this.numSkills = skill;
        this.numConcepts = concept;
        this.lambda_s = lambda_s;
        this.lambda_t = lambda_t;
        this.lambda_q = lambda_q;
        this.lambda_l = lambda_l;
        this.lambda_d = lambda_d;
        this.lambda_bias = lambda_bias;
        this.penality_weight = penality_weight;
        this.markovian = markovian;
        this.lectureTradeOff = lectureTradeOff;
        this.discussionTradeOff = discussionTradeOff;
        this.lr = lr;
        this.iter = iter;
        this.tol = 0.001;
        this.top_k = top_k;
        this.startAttempt = startAttempt;
        this.endAttempt = endAttempt;
        this.metrics = metrics;
        this.logfile = logfile;

        this.userData = userData;
        this.trainSet = trainSet;
        this.testSet = testSet;
        this.testUsers = testUsers;
        this.testUserRecords = testUserRecords;
        this.testUserTestingPeriod = testUserTestingPeriod;
        this.testUserTrainingPeriod = testUserTrainingPeriod;
        this.questionScoresMap = questionScoresMap;
        this.nextQuestionsMap = nextQuestionsMap;
    }

    public void extractTrainTestUsers(String dataStr, Integer course){
        //read the whole dataset and extract the testing users who have all attempts
        LinkedHashMap<String, String> quizMap = new LinkedHashMap<>();
        LinkedHashMap<String, String> userMap = new LinkedHashMap<>();
        ArrayList<ArrayList<Number>> allData = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(String.format("data/%s/%s/quiz_item.txt", dataStr, course.toString()))))
        {
            String line;
            while((line=reader.readLine()) != null){
                String[] fields = line.trim().split(" ");
                quizMap.put(fields[0], fields[1]);
            }
            reader.close();
        } 
        catch (FileNotFoundException e) 
        {
            e.printStackTrace();
        } 
        catch (IOException e) 
        {
            e.printStackTrace();
        } 

        try (BufferedReader reader = new BufferedReader(new FileReader(String.format("data/%s/%s/user.txt", dataStr, course.toString()))))
        {
            String line;
            while((line=reader.readLine()) != null){
                String[] fields = line.trim().split(" ");
                userMap.put(fields[0], fields[1]);
            }
            reader.close();
        } 
        catch (FileNotFoundException e) 
        {
            e.printStackTrace();
        } 
        catch (IOException e) 
        {
            e.printStackTrace();
        } 

        //JSON parser object to parse read file
        JSONParser jsonParser = new JSONParser();
        JSONObject data = null;
        
        // Trys reading the data from a json file
        try (FileReader reader = new FileReader(String.format("data/%s/%s/1_train_test.json", dataStr, course.toString())))
        {
            //Read JSON file
            data = (JSONObject) jsonParser.parse(reader);
            reader.close();
        } 
        catch (FileNotFoundException e) 
        {
            e.printStackTrace();
        } 
        catch (IOException e) 
        {
            e.printStackTrace();
        } 
        catch (ParseException e) 
        {
            e.printStackTrace();
        }
        
        System.out.println(data.toJSONString());

        JSONArray dataTrain = (JSONArray) data.get("train");
        for(int i=0; i<dataTrain.size(); i++){
            JSONArray innerJSONArray = (JSONArray) dataTrain.get(i);
            ArrayList<Number> innerArrayList = new ArrayList<>();

            Integer student = (int) innerJSONArray.get(0);
            innerArrayList.add(student);
            Integer attempt = (int) innerJSONArray.get(3);
            innerArrayList.add(attempt);
            Integer question = (int) innerJSONArray.get(1);
            innerArrayList.add(question);
            Double obs = (double) innerJSONArray.get(2);
            innerArrayList.add(obs);
            Integer resource = (int) innerJSONArray.get(4);
            innerArrayList.add(resource);

            allData.add(innerArrayList);
        }

        JSONArray dataTest = (JSONArray) data.get("test");
        for(int i=0; i<dataTest.size(); i++){
            JSONArray innerJSONArray = (JSONArray) dataTest.get(i);
            ArrayList<Number> innerArrayList = new ArrayList<>();

            Integer student = (int) innerJSONArray.get(0);
            innerArrayList.add(student);
            Integer attempt = (int) innerJSONArray.get(3);
            innerArrayList.add(attempt);
            Integer question = (int) innerJSONArray.get(1);
            innerArrayList.add(question);
            Double obs = (double) innerJSONArray.get(2);
            innerArrayList.add(obs);
            Integer resource = (int) innerJSONArray.get(4);
            innerArrayList.add(resource);

            allData.add(innerArrayList);
        }

        // map student to a list of questions and obs
        LinkedHashMap<Integer, ArrayList<Integer>> studentQuestionMap = new LinkedHashMap<>();
        LinkedHashMap<Integer, ArrayList<Double>> studentObsMap = new LinkedHashMap<>();
        for(ArrayList<Number> innerArray : allData){
            Integer student = (Integer) innerArray.get(0);
            Integer question = (Integer) innerArray.get(1);
            Integer attempt = (Integer) innerArray.get(2);
            Double obs = (Double) innerArray.get(3);
            Integer resource = (Integer) innerArray.get(4);

            if(!studentQuestionMap.containsKey(student)){
                studentQuestionMap.put(student, new ArrayList<>());
                studentObsMap.put(student, new ArrayList<>());
            }

            studentQuestionMap.get(student).add(question);
            studentObsMap.get(student).add(obs);
        }   

        // filter out students who have attempts less than 20
        for(Integer student : studentQuestionMap.keySet()){
            if(studentQuestionMap.get(student).size() != 25){
                studentQuestionMap.remove(student);
            }
        }

        System.out.println(studentQuestionMap.size());
        
        for(Integer student : studentQuestionMap.keySet()) {
        	System.out.println(student + ": " + studentQuestionMap.get(student));
        }
        System.out.println();
        
        Number[] min_student = {0, 10000.0};
        Number[] max_student = {0, 0.0};
        for(Integer student : studentQuestionMap.keySet()) {
        	Double avg = meanI(studentQuestionMap.get(student));
        	if (avg < (Double) min_student[1]) {
        		min_student[0] = student;
        		min_student[1] = avg;
        	}
        	if(avg > (Double) max_student[1]) {
        		max_student[0] = student;
        		max_student[1] = avg;
        	}
        }
        
        System.out.println("max: " + max_student[0] + ", " + max_student[1]);
        System.out.println("min: " + min_student[0] + ", " + min_student[1]);

        JSONArray train_set = new JSONArray();
        JSONArray test_set = new JSONArray();
        for(ArrayList<Number> innerArray : allData){
            JSONArray set = new JSONArray();

            Integer student = (Integer) innerArray.get(0);
            set.add(student);
            Integer question = (Integer) innerArray.get(1);
            set.add(question);
            Integer attempt = (Integer) innerArray.get(2);
            set.add(attempt);
            Double obs = (Double) innerArray.get(3);
            set.add(obs);
            Integer resource = (Integer) innerArray.get(4);
            set.add(resource);


            if(studentQuestionMap.containsKey(student) && attempt >= 10){
                test_set.add(set);
            }
            else{
                train_set.add(set);
            }
        }
        
        JSONObject newData = new JSONObject();
        newData.put("num_attempts", 25);
        newData.put("num_users", 459);
        newData.put("num_quizs", 18);
        newData.put("train", train_set);
        newData.put("test", test_set);

        System.out.println(test_set.toJSONString());

        //Write JSON file
        try (FileWriter file = new FileWriter(String.format("data/%s/%s/0_train_test.json", dataStr, course.toString()))) {
 
            file.write(newData.toJSONString());
            file.flush();
            file.close();
 
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static boolean isIncreasing(ArrayList<Double> arr)
    {
        for(int i=1; i<arr.size();i++)
        {
            if(arr.get(i-1)>arr.get(i))
                return false;
        }
        return true;
    }

    private double mean(ArrayList<Double> l) 
    {
        Double sum = 0.0;
        if(!l.isEmpty()) {
          for (Double d : l) {
              sum += d;
          }
          return sum / l.size();
        }
        return sum;
    }
    
    private double meanI(ArrayList<Integer> l) 
    {
        Double sum = 0.0;
        if(!l.isEmpty()) {
          for (Integer i : l) {
        	  Double d = i.doubleValue();
              sum += d;
          }
          return sum / l.size();
        }
        return sum;
    }

    private double std(ArrayList<Double> l)
    {
        double mean = mean(l);
        double std = 0.0;

        for(double num: l) {
            std += Math.pow(num - mean, 2);
        }

        return Math.sqrt(std/l.size());
    }
}