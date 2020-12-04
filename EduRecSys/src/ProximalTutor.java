import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.lang.Math;

public class ProximalTutor {
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
	LinkedHashMap<Integer, ArrayList<Double>> testUserTrainingPeriod;
	LinkedHashMap<Integer, ArrayList<Double>> testUserTestingPeriod;
	LinkedHashMap<Integer, ArrayList<Double>> questionScoresMap;
	LinkedHashMap<Integer, LinkedHashMap<Integer, Integer>> nextQuestionsMap;

	LinkedHashMap<Integer, Boolean> rareQuestions;

	ArrayList<Double> testObsList;
	ArrayList<Double> testPredList;

	Boolean useBiasT;
	Boolean useGlobalBias;
	Boolean exactPenalty;

	Boolean binarizedQuestion;
	Boolean binarizedLecture;
	Boolean binarizedDiscussion;

	LinkedHashMap<Integer, LinkedHashMap<Integer, ArrayList<Double>>> historicalRecords;

	ArrayList<ArrayList<Number>> trainDataMarkovian;

	Double[][] S;
	Double[][][] T;
	Double[][] Q;
	Double[][] L;
	Double[][] D;
	Double[] bias_s;
	Double[] bias_t;
	Double[] bias_q;
	Double[] bias_l;
	Double[] bias_d;
	Double globalBias;

	LinkedHashMap<Integer, Integer> currentQuestion;
	LinkedHashMap<Object, Object> currentStates;
	LinkedHashMap<Integer, Double> currentScore;

	LinkedHashMap<Integer, ArrayList<Double>> recommendation;

	LinkedHashMap<Integer, Integer> recMap;
	
	LinkedHashMap<Integer, LinkedHashMap<Integer, LinkedHashMap<Integer, Integer>>> confidenceMap;
	
	Integer total_hit;
	Integer total_miss;

	public ProximalTutor(DataHelper config) throws Exception {
		this.dataStr = config.dataStr;
		this.views = config.views;
		this.numUsers = config.numUsers;
		this.numAttempts = config.numAttempts;
		this.numQuestions = config.numQuestions;
		this.numLectures = config.numLectures;
		this.numDiscussions = config.numDiscussions;
		this.numSkills = config.numSkills;
		this.numConcepts = config.numConcepts;
		this.lambda_s = config.lambda_s;
		this.lambda_t = config.lambda_t;
		this.lambda_q = config.lambda_q;
		this.lambda_l = config.lambda_l;
		this.lambda_d = config.lambda_d;
		this.lambda_bias = config.lambda_bias;
		this.penality_weight = config.penality_weight;
		this.markovian = config.markovian;
		this.lectureTradeOff = config.lectureTradeOff;
		this.discussionTradeOff = config.discussionTradeOff;
		this.lr = config.lr;
		this.iter = config.iter;
		this.tol = config.tol;
		this.top_k = config.top_k;
		this.startAttempt = config.startAttempt;
		this.endAttempt = config.endAttempt;
		this.metrics = config.metrics;
		this.logfile = config.logfile;
		this.userData = config.userData;
		this.trainSet = config.trainSet;
		this.testSet = config.testSet;
		this.testUsers = config.testUsers;
		this.testUserRecords = config.testUserRecords;
		this.testUserTestingPeriod = config.testUserTestingPeriod;
		this.testUserTrainingPeriod = config.testUserTrainingPeriod;
		this.questionScoresMap = config.questionScoresMap;
		this.nextQuestionsMap = config.nextQuestionsMap;

		// we need to check the rare questions
		this.rareQuestions = new LinkedHashMap<>();
		for (Integer question : this.questionScoresMap.keySet()) {
			if (this.questionScoresMap.get(question).size() < 5) {
				this.rareQuestions.put(question, true);
			}
		}

		this.testObsList = new ArrayList<>();
		this.testPredList = new ArrayList<>();

		this.useBiasT = false;
		this.useGlobalBias = true;
		this.exactPenalty = false;

		this.binarizedQuestion = true;
		this.binarizedLecture = true;
		this.binarizedDiscussion = true;

		// used to dynamically store the logged record based on key = question
		this.historicalRecords = new LinkedHashMap<>();
		this.total_hit = 0;
		this.total_miss = 0;

		// get extended training data, since we have the markovian assumption,
		// which are stored in train_data_dict and train_data_markovian_dict
		LinkedHashMap<String, Double> trainDataMap = new LinkedHashMap<>();
		for (ArrayList<Number> innerList : this.trainSet) {
			ArrayList<String> key = new ArrayList<String>();

			String student = ((Integer) innerList.get(0)).toString();
			key.add(student);
			String attempt = ((Integer) innerList.get(1)).toString();
			key.add(attempt);
			String question = ((Integer) innerList.get(2)).toString();
			key.add(question);
			Double obs = (Double) innerList.get(3);
			String resource = ((Integer) innerList.get(4)).toString();
			key.add(resource);

			String keyStr = String.join(",", key);
			if (!trainDataMap.containsKey(keyStr)) {
				trainDataMap.put(keyStr, obs);
			}
		}

		this.trainDataMarkovian = new ArrayList<>();
		LinkedHashMap<String, Boolean> trainDataMarkovianMap = new LinkedHashMap<>();
		for (ArrayList<Number> innerList : this.trainSet) {
			Integer student = (Integer) innerList.get(0);
			Integer attempt = (Integer) innerList.get(1);
			Integer question = (Integer) innerList.get(2);
			Double obs = (Double) innerList.get(3);
			Integer resource = (Integer) innerList.get(4);

			Integer upper_steps = Math.min(this.numAttempts, attempt + this.markovian + 1);
			for (Integer j = attempt + 1; j < upper_steps; j++) {
				ArrayList<String> keyList = new ArrayList<>();
				ArrayList<Number> numList = new ArrayList<>();
				keyList.add(student.toString());
				keyList.add(j.toString());
				keyList.add(question.toString());
				keyList.add(resource.toString());

				numList.add(student);
				numList.add(j);
				numList.add(question);
				numList.add(resource);

				String key = String.join(",", keyList);
				if (!trainDataMap.containsKey(key)) {
					if (!trainDataMarkovianMap.containsKey(key)) {
						trainDataMarkovianMap.put(key, true);
						this.trainDataMarkovian.add(numList);
					}
				}
			}
		}

		if (this.views.charAt(0) == '1') {
			this.S = randomSample(this.numUsers, this.numSkills);
			this.T = randomSample(this.numSkills, this.numAttempts, this.numConcepts);
			this.Q = randomSample(this.numConcepts, this.numQuestions);
			this.L = zeros(this.numConcepts, this.numLectures);
			this.D = zeros(this.numConcepts, this.numDiscussions);
			this.bias_s = zeros(this.numUsers);
			this.bias_t = zeros(this.numAttempts);
			this.bias_q = zeros(this.numQuestions);
			this.bias_l = zeros(this.numLectures);
			this.bias_d = zeros(this.numDiscussions);
			this.globalBias = meanAxis0(this.trainSet)[3];
		} else {
			throw new Exception("Attribute Error");
		}

		if (this.views.charAt(1) == '1') {
			this.L = randomSample(this.numConcepts, this.numLectures);
		}
		if (this.views.charAt(1) == '1') {
			this.D = randomSample(this.numConcepts, this.numDiscussions);
		}

		// get the latest questions for all users from the training data
		this.currentQuestion = new LinkedHashMap<>();
		this.currentStates = new LinkedHashMap<>();
		this.currentScore = new LinkedHashMap<>();
		LinkedHashMap<Integer, ArrayList<Number>> records = new LinkedHashMap<>();
		for (ArrayList<Number> innerList : this.trainSet) {
			ArrayList<Number> record = new ArrayList<>();

			Integer student = (Integer) innerList.get(0);
			Integer attempt = (Integer) innerList.get(1);
			record.add(attempt);
			Integer question = (Integer) innerList.get(2);
			record.add(question);
			Double outcome = (Double) innerList.get(3);
			record.add(outcome);
			Integer resource = (Integer) innerList.get(4);
			record.add(resource);

			if (!records.containsKey(student)) {
				records.put(student, record);
			} else {
				Integer max_attempt = (Integer) records.get(student).get(0);
				if (attempt > max_attempt) {
					records.get(student).set(0, attempt);
				}
			}
		}

		for (Integer user : this.userData.keySet()) {
			ArrayList<ArrayList<Number>> allRecords = this.userData.get(user);
			if (this.testUsers.containsKey(user)) {
				if (!historicalRecords.containsKey(user)) {
					this.historicalRecords.put(user, new LinkedHashMap<>());
				}

				for (ArrayList<Number> innerList : allRecords) {
					Integer student = (Integer) innerList.get(0);
					Integer attempt = (Integer) innerList.get(1);
					Integer question = (Integer) innerList.get(2);
					Double obs = (Double) innerList.get(3);
					Integer resource = (Integer) innerList.get(4);

					if (attempt < this.startAttempt) {
						if (!this.historicalRecords.get(user).containsKey(question)) {
							this.historicalRecords.get(user).put(question, new ArrayList<>());
						}
						this.historicalRecords.get(user).get(question).add(obs);
					}
				}
			}
		}

		for (Integer student : this.testUsers.keySet()) {
			// current question can be viewed as current state
			this.currentQuestion.put(student, (Integer) records.get(student).get(1));
			this.currentScore.put(student, (Double) records.get(student).get(2));
		}

		// initialize the recommendation performance as 0 for each test user
		this.recommendation = new LinkedHashMap<>();
		for (Integer student : testUsers.keySet()) {
			this.recommendation.put(student, new ArrayList<>());
		}

		this.recMap = new LinkedHashMap<>();
	}

	private Double getQuestionPrediction(Integer student, Integer attempt, Integer question) throws Exception {
		/**
		 * predict value at tensor Y[attempt, student, question]
		 * 
		 * @param attempt:  attempt index
		 * @param student:  student index
		 * @param question: question index
		 * @return: predicted value of tensor Y[attempt, student, question]
		 */

		Double pred = dot(dot(this.S[student], slice(this.T, attempt)), slice(this.Q, question));
		if (this.useBiasT) {
			if (this.useGlobalBias) {
				pred += this.bias_s[student] + this.bias_t[attempt] + this.bias_q[question] + this.globalBias;
			} else {
				pred += this.bias_s[student] + this.bias_t[attempt] + this.bias_q[question];
			}
		} else {
			if (this.useGlobalBias) {
				pred += this.bias_s[student] + this.bias_q[question] + this.globalBias;
			} else {
				pred += this.bias_s[student] + this.bias_q[question];
			}
		}

		if (this.binarizedQuestion) {
			pred = sigmoid(pred, false);
		}
		return pred;
	}

	private Double getLecturePrediction(Integer student, Integer attempt, Integer lecture) throws Exception {
		/**
		 * predict value at tensor Y[attempt, student, lecture]
		 * 
		 * @param attempt: attempt index
		 * @param student: student index
		 * @param lecture: lecture index
		 * @return: predicted value of tensor Y[attempt, student, lecture]
		 */

		Double pred = dot(dot(this.S[student], slice(this.T, attempt)), slice(this.L, lecture));
		if (this.useBiasT) {
			if (this.useGlobalBias) {
				pred += this.bias_s[student] + this.bias_t[attempt] + this.bias_l[lecture] + this.globalBias;
			} else {
				pred += this.bias_s[student] + this.bias_t[attempt] + this.bias_l[lecture];
			}
		} else {
			if (this.useGlobalBias) {
				pred += this.bias_s[student] + this.bias_l[lecture] + this.globalBias;
			} else {
				pred += this.bias_s[student] + this.bias_l[lecture];
			}
		}

		if (this.binarizedQuestion) {
			pred = sigmoid(pred, false);
		}

		return pred;
	}

	private Double getDiscussionPrediction(Integer student, Integer attempt, Integer discussion) throws Exception {
		/**
		 * predict value at tensor Y[attempt, student, discussion]
		 * 
		 * @param attempt:    attempt index
		 * @param student:    student index
		 * @param discussion: discussion index
		 * @return: predicted value of tensor Y[attempt, student, discussion]
		 */

		Double pred = dot(dot(this.S[student], slice(this.T, attempt)), slice(this.D, discussion));
		if (this.useBiasT) {
			if (this.useGlobalBias) {
				pred += this.bias_s[student] + this.bias_t[attempt] + this.bias_d[discussion] + this.globalBias;
			} else {
				pred += this.bias_s[student] + this.bias_t[attempt] + this.bias_d[discussion];
			}
		} else {
			if (this.useGlobalBias) {
				pred += this.bias_s[student] + this.bias_d[discussion] + this.globalBias;
			} else {
				pred += this.bias_s[student] + this.bias_d[discussion];
			}
		}

		if (this.binarizedQuestion) {
			pred = sigmoid(pred, false);
		}

		return pred;
	}

	public Double[] getLoss() throws Exception {
		/**
		 * override the function in super class compute the loss, which is RMSE of
		 * observed records + regularization + penalty of temporal non-smoothness
		 * 
		 * @return: loss
		 */

		Double loss = 0.0;
		Double squareLoss = 0.0;
		Double regBias = 0.0;

		Double squareLoss_q = 0.0;
		Double squareLoss_l = 0.0;
		Double squareLoss_d = 0.0;

		Double q_count = 0.0;
		Double l_count = 0.0;
		Double d_count = 0.0;

		Double pred;

		for (ArrayList<Number> innerList : this.trainSet) {
			Integer student = (Integer) innerList.get(0);
			Integer attempt = (Integer) innerList.get(1);
			Integer question = (Integer) innerList.get(2);
			Double obs = (Double) innerList.get(3);
			Integer resource = (Integer) innerList.get(4);

			if (resource == 0) {
				pred = getQuestionPrediction(student, attempt, question);
				squareLoss_q += Math.pow((obs - pred), 2);
				q_count++;
			} else if (resource == 1) {
				pred = getLecturePrediction(student, attempt, question);
				squareLoss_l += Math.pow((obs - pred), 2);
				l_count++;
			} else if (resource == 2) {
				pred = getDiscussionPrediction(student, attempt, question);
				squareLoss_d += Math.pow((obs - pred), 2);
				d_count++;
			}
		}

		squareLoss = squareLoss_q + this.lectureTradeOff * squareLoss_l + this.discussionTradeOff * squareLoss_d;

		// regularzation of student data
		Double reg_S = norm(this.S);
		Double reg_T = norm(this.T);
		Double reg_Q = norm(this.Q);
		Double reg_L = norm(this.L);
		Double reg_D = norm(this.D);

		Double reg_features = this.lambda_s * reg_S + this.lambda_q * reg_Q + this.lambda_t * reg_T
				+ this.lambda_l * reg_L + this.lambda_d * reg_D;
		Double q_rmse = (q_count != 0) ? Math.sqrt(squareLoss_q / q_count) : 0;
		Double l_rmse = (l_count != 0) ? Math.sqrt(this.lectureTradeOff * squareLoss_l / l_count) : 0;
		Double d_rmse = (d_count != 0) ? Math.sqrt(this.discussionTradeOff * squareLoss_d / d_count) : 0;

		Double reg_bias = 0.0;
		if (this.lambda_bias != null) {
			if (this.useBiasT) {
				reg_bias = this.lambda_bias * (norm(this.bias_s) + norm(this.bias_t) + norm(this.bias_q)
						+ norm(this.bias_l) + norm(this.bias_d));
			} else {
				reg_bias = this.lambda_bias
						* (norm(this.bias_s) + norm(this.bias_q) + norm(this.bias_l) + norm(this.bias_d));
			}
		}

		Double penality = getPenalty();
		loss = squareLoss + reg_features + reg_bias + penality;
		Double[] losses = { loss, q_count, q_rmse, l_rmse, d_rmse, penality, reg_features, reg_bias };
		return losses;
	}

	public Double getPenalty() throws Exception {
		/**
		 * compute the penalty on the observations, we want all attempts before the obs
		 * has smaller score, and the score after obs should be greater. we use sigmoid
		 * to set the penalty between 0 and 1 if knowledge at current attempt >> prev
		 * attempt, then diff is large, that mean sigmoid(diff) is large and close to
		 * 1., so penalty is a very small negative number since we aim to minimize the
		 * objective = loss + penalty, the smaller penalty is better
		 */

		Double penality = 0.0;
		for (ArrayList<Number> innerList : this.trainSet) {
			Integer student = (Integer) innerList.get(0);
			Integer attempt = (Integer) innerList.get(1);
			Integer index = (Integer) innerList.get(2);
			Double obs = (Double) innerList.get(3);
			Integer resource = (Integer) innerList.get(4);

			if (attempt >= 1 && resource == 0) {
				Double[][] gap = matSub(slice(this.T, attempt), slice(this.T, attempt - 1));
				Double[] knowledge_gap = dot(this.S[student], gap);
				Double penality_val;
				if (this.exactPenalty) {
					for (int i = 0; i < knowledge_gap.length; i++) {
						if (knowledge_gap[i] > 0.0) {
							knowledge_gap[i] = 0.0;
						}
					}

					penality_val = -1.0 * dot(knowledge_gap, slice(this.Q, index));
				} else {
					Double diff = dot(knowledge_gap, slice(this.Q, index));
					penality_val = -1.0 * Math.log(sigmoid(diff, false));
				}
				penality += this.penality_weight * penality_val;
			}
		}

		for (ArrayList<Number> innerList : this.trainDataMarkovian) {
			Integer student = (Integer) innerList.get(0);
			Integer attempt = (Integer) innerList.get(1);
			Integer index = (Integer) innerList.get(2);
			Integer resource = (Integer) innerList.get(3);

			if (attempt >= 1 && resource == 0) {
				Double[][] gap = matSub(slice(this.T, attempt), slice(this.T, attempt - 1));
				Double[] knowledge_gap = dot(this.S[student], gap);
				Double penality_val;
				if (this.exactPenalty) {
					for (int i = 0; i < knowledge_gap.length; i++) {
						if (knowledge_gap[i] > 0.0) {
							knowledge_gap[i] = 0.0;
						}
					}

					penality_val = -1.0 * dot(knowledge_gap, slice(this.Q, index));
				} else {
					Double diff = dot(knowledge_gap, slice(this.Q, index));
					penality_val = -1.0 * Math.log(sigmoid(diff, false));
				}
				penality += this.penality_weight * penality_val;
			}
		}

		return penality;
	}

	public Double[] gradS_k(Integer student, Integer attempt, Integer index, Double obs, Integer resource) throws Exception {
		/**
		 * note that the penalty is actually next score is larger than previous score
		 * instead of saying next knowledge state is higher than previous knowledge
		 * state
		 */

		Double[] grad = zeros(this.S[student].length);
		Double pred;
		if (obs != null) {
			if (resource == 0) {
				pred = getQuestionPrediction(student, attempt, index);
				if (this.binarizedQuestion) {
					grad = scalarMulVector(-2.0 * (obs - pred) * pred * (1.0 - pred),
							dot(slice(this.T, attempt), slice(this.Q, index)));
				} else {
					grad = scalarMulVector((-2.0 * (obs - pred)), dot(slice(this.T, attempt), slice(this.Q, index)));
				}
			} else if (resource == 1) {
				pred = getLecturePrediction(student, attempt, index);
				if (this.binarizedLecture) {
					grad = scalarMulVector(-2.0 * this.lectureTradeOff * (obs - pred) * pred * (1.0 - pred),
							dot(slice(this.T, attempt), slice(this.L, index)));
				} else {
					grad = scalarMulVector((-2.0 * this.lectureTradeOff * (obs - pred)),
							dot(slice(this.T, attempt), slice(this.L, index)));
				}
			} else if (resource == 2) {
				pred = getDiscussionPrediction(student, attempt, index);
				if (this.binarizedDiscussion) {
					grad = scalarMulVector(-2.0 * this.discussionTradeOff * (obs - pred) * pred * (1.0 - pred),
							dot(slice(this.T, attempt), slice(this.D, index)));
				} else {
					grad = scalarMulVector((-2.0 * this.discussionTradeOff * (obs - pred)),
							dot(slice(this.T, attempt), slice(this.D, index)));
				}
			}
		}

		grad = addVectors(grad, scalarMulVector((2.0 * this.lambda_s), this.S[student]));

		if (resource == 0) {
			Double[][] diff;
			if (attempt == 0) {
				diff = matSub(slice(this.T, attempt + 1), slice(this.T, attempt));
			} else if (attempt == this.numAttempts - 1) {
				diff = matSub(slice(this.T, attempt), slice(this.T, attempt - 1));
			} else {
				diff = matSub(slice(this.T, attempt), slice(this.T, attempt - 1));
				diff = matAdd(diff, matSub(slice(this.T, attempt + 1), slice(this.T, attempt)));
			}

			Double[] TQDiff;
			if (this.exactPenalty) {
				TQDiff = dot(diff, slice(this.Q, index));
				for (int i = 0; i < TQDiff.length; i++) {
					if (TQDiff[i] > 0.0) {
						TQDiff[i] = 0.0;
					}
				}

				grad = addVectors(grad, scalarMulVector(this.penality_weight * (-1.0), TQDiff));
			} else {
				TQDiff = dot(diff, slice(this.Q, index));
				Double val = dot(this.S[student], TQDiff);
				grad = addVectors(grad,
						scalarMulVector(this.penality_weight
								* (1.0 / sigmoid(val, false) * sigmoid(val, false) * (1.0 - sigmoid(val, false))),
								TQDiff));
			}
		}

		return grad;
	}

	public Double[][] gradT_ij(Integer student, Integer attempt, Integer index, Double obs, Integer resource) throws Exception {
		/**
		 * compute the gradient of loss w.r.t a specific student j's knowledge at a
		 * specific attempt i: T_{i,j,:}
		 */

		Double[][] temp = slice(this.T, attempt);
		Double[][] grad = zeros(temp.length, temp[0].length);
		if (obs != null) {
			Double pred;
			if (resource == 0) {
				pred = getQuestionPrediction(student, attempt, index);
				if (this.binarizedQuestion) {
					grad = scalarMulMat(-2.0 * (obs - pred) * pred * (1.0 - pred),
							outer(this.S[student], slice(this.Q, index)));
				} else {
					grad = scalarMulMat((-2.0 * (obs - pred)), outer(this.S[student], slice(this.Q, index)));
				}
			} else if (resource == 1) {
				pred = getLecturePrediction(student, attempt, index);
				if (this.binarizedLecture) {
					grad = scalarMulMat(-2.0 * this.lectureTradeOff * (obs - pred) * pred * (1.0 - pred),
							outer(this.S[student], slice(this.L, index)));
				} else {
					grad = scalarMulMat((-2.0 * this.lectureTradeOff * (obs - pred)),
							outer(this.S[student], slice(this.L, index)));
				}
			} else if (resource == 2) {
				pred = getDiscussionPrediction(student, attempt, index);
				if (this.binarizedDiscussion) {
					grad = scalarMulMat(-2.0 * this.discussionTradeOff * (obs - pred) * pred * (1.0 - pred),
							outer(this.S[student], slice(this.D, index)));
				} else {
					grad = scalarMulMat((-2.0 * this.discussionTradeOff * (obs - pred)),
							outer(this.S[student], slice(this.D, index)));
				}
			}
		}

		grad = matAdd(grad, scalarMulMat(2.0 * this.lambda_t, slice(this.T, attempt)));

		if (resource == 0) {
			Double[][] diff;
			if (attempt == 0) {
				diff = matSub(slice(this.T, attempt + 1), slice(this.T, attempt));
				if (this.exactPenalty) {
					for (int i = 0; i < diff.length; i++) {
						for (int j = 0; j < diff.length; j++) {
							if (diff[i][j] > 0.0) {
								diff[i][j] = 0.0;
							}
						}
					}

					Double penality_val = (-1.0) * dot(dot(this.S[student], diff), slice(this.Q, index));
					grad = matPlusScalar(grad, this.penality_weight * penality_val * (-1.0));
				} else {
					Double val = dot(dot(this.S[student], diff), slice(this.Q, index));
					grad = matAdd(grad, scalarMulMat((-1.0) * this.penality_weight * (1.0 / sigmoid(val, false) * sigmoid(val, false) * (1.0 - sigmoid(val, false)) * (-1.0)), outer(this.S[student], slice(this.Q, index))));
				}
			} else if (attempt == this.numAttempts - 1) {
				diff = matSub(slice(this.T, attempt), slice(this.T, attempt - 1));
				if (this.exactPenalty) {
					for (int i = 0; i < diff.length; i++) {
						for (int j = 0; j < diff.length; j++) {
							if (diff[i][j] > 0.0) {
								diff[i][j] = 0.0;
							}
						}
					}

					Double penality_val = (-1.0) * dot(dot(this.S[student], diff), slice(this.Q, index));
					grad = matPlusScalar(grad, this.penality_weight * penality_val);
				} else {
					Double val = dot(dot(this.S[student], diff), slice(this.Q, index));
					grad = matAdd(grad, scalarMulMat((-1.0) * this.penality_weight * (1.0 / sigmoid(val, false) * sigmoid(val, false) * (1.0 - sigmoid(val, false))), outer(this.S[student], slice(this.Q, index))));
				}
			} else {
				if (this.exactPenalty) {
					diff = matSub(slice(this.T, attempt), slice(this.T, attempt - 1));
					for (int i = 0; i < diff.length; i++) {
						for (int j = 0; j < diff.length; j++) {
							if (diff[i][j] > 0.0) {
								diff[i][j] = 0.0;
							}
						}
					}

					Double penality_val = (-1.0) * dot(dot(this.S[student], diff), slice(this.Q, index));
					grad = matPlusScalar(grad, this.penality_weight * penality_val);

					diff = matSub(slice(this.T, attempt + 1), slice(this.T, attempt));
					for (int i = 0; i < diff.length; i++) {
						for (int j = 0; j < diff.length; j++) {
							if (diff[i][j] > 0.0) {
								diff[i][j] = 0.0;
							}
						}
					}

					penality_val = (-1.0) * dot(dot(this.S[student], diff), slice(this.Q, index));
					grad = matPlusScalar(grad, this.penality_weight * penality_val * (-1.0));
				} else {
					diff = matSub(slice(this.T, attempt), slice(this.T, attempt - 1));
					Double val = dot(dot(this.S[student], diff), slice(this.Q, index));
					grad = matAdd(grad,scalarMulMat((-1.0) * this.penality_weight * (1.0 / sigmoid(val, false) * sigmoid(val, false) * (1.0 - sigmoid(val, false))), outer(this.S[student], slice(this.Q, index))));

					diff = matSub(slice(this.T, attempt + 1), slice(this.T, attempt));
					val = dot(dot(this.S[student], diff), slice(this.Q, index));
					grad = matAdd(grad,scalarMulMat((-1.0) * this.penality_weight * (1.0 / sigmoid(val, false) * sigmoid(val, false) * (1.0 - sigmoid(val, false)) * (-1.0)), outer(this.S[student], slice(this.Q, index))));
				}
			}
		}

		return grad;
	}

	public Double[] gradQ_k(Integer student, Integer attempt, Integer question, Double obs) throws Exception {
		/**
		 * compute the gradient of loss w.r.t a specific concept-question association of
		 * a question in Q-matrix,
		 */

		Double[] grad = zeros(slice(this.Q, question).length);
		if (obs != null) {
			Double pred = getQuestionPrediction(student, attempt, question);
			if (this.binarizedQuestion) {
				grad = scalarMulVector(-2.0 * (obs - pred) * pred * (1 - pred),
						dot(this.S[student], slice(this.T, attempt)));
			} else {
				grad = scalarMulVector(-2.0 * (obs - pred), dot(this.S[student], slice(this.T, attempt)));
			}
		}

		grad = addVectors(grad, scalarMulVector(-2.0 * this.lambda_q, slice(this.Q, question)));

		Double[][] diff;
		if (attempt == 0) {
			diff = matSub(slice(this.T, attempt + 1), slice(this.T, attempt));
		} else if (attempt == this.numAttempts - 1) {
			diff = matSub(slice(this.T, attempt), slice(this.T, attempt - 1));
		} else {
			diff = matSub(slice(this.T, attempt), slice(this.T, attempt - 1));
			diff = matAdd(diff, matSub(slice(this.T, attempt + 1), slice(this.T, attempt)));
		}

		Double[] knowledge_gap = dot(this.S[student], diff);

		if (this.exactPenalty) {
			for (int i = 0; i < knowledge_gap.length; i++) {
				if (knowledge_gap[i] > 0) {
					knowledge_gap[i] = 0.0;
				}
			}

			grad = addVectors(grad, scalarMulVector(-1.0 * this.penality_weight, knowledge_gap));
		} else {
			Double val = dot(knowledge_gap, slice(this.Q, question));
			grad = addVectors(grad, scalarMulVector(-1.0 * this.penality_weight* (1.0 / sigmoid(val, false) * sigmoid(val, false) * (1.0 - sigmoid(val, false))),knowledge_gap));
		}
		return grad;
	}

	public Double[] gradL_k(Integer student, Integer attempt, Integer lecture, Double obs) throws Exception {
		/**
		 * compute the gradient of loss w.r.t a specific concept-question association of
		 * a question in Q-matrix,
		 */

		Double[] grad = zeros(slice(this.L, lecture).length);
		if (obs != null) {
			Double pred = getLecturePrediction(student, attempt, lecture);
			if (this.binarizedLecture) {
				grad = scalarMulVector(-2.0 * this.lectureTradeOff * (obs - pred) * pred * (1.0 - pred),
						dot(this.S[student], slice(this.T, attempt)));
			} else {
				grad = scalarMulVector(-2.0 * this.lectureTradeOff * (obs - pred),
						dot(this.S[student], slice(this.T, attempt)));
			}
		}

		grad = addVectors(grad, scalarMulVector(-2.0 * this.lambda_d, slice(this.L, lecture)));
		return grad;
	}

	public Double[] gradD_k(Integer student, Integer attempt, Integer discussion, Double obs) throws Exception {
		/**
		 * compute the gradient of loss w.r.t a specific concept-question association of
		 * a question in Q-matrix,
		 */

		Double[] grad = zeros(slice(this.D, discussion).length);
		if (obs != null) {
			Double pred = getDiscussionPrediction(student, attempt, discussion);
			if (this.binarizedDiscussion) {
				grad = scalarMulVector(-2.0 * this.discussionTradeOff * (obs - pred) * pred * (1.0 - pred),
						dot(this.S[student], slice(this.T, attempt)));
			} else {
				grad = scalarMulVector(-2.0 * this.discussionTradeOff * (obs - pred),
						dot(this.S[student], slice(this.T, attempt)));
			}
		}

		grad = addVectors(grad, scalarMulVector(2.0 * this.lambda_d, slice(this.D, discussion)));
		return grad;
	}

	public Double gradBias_s(Integer student, Integer attempt, Integer material, Double obs, Integer resource) throws Exception {
		/**
		 * compute the gradient of loss w.r.t a specific bias_s
		 */

		Double grad = 0.0;
		if (obs != null) {
			Double pred;
			if (resource == 0) {
				pred = getQuestionPrediction(student, attempt, material);
				if (this.binarizedQuestion) {
					grad -= 2.0 * (obs - pred) * pred * (1.0 - pred);
				} else {
					grad -= 2.0 * (obs - pred);
				}
			} else if (resource == 1) {
				pred = getLecturePrediction(student, attempt, material);
				if (this.binarizedLecture) {
					grad -= 2.0 * this.lectureTradeOff * (obs - pred) * pred * (1.0 - pred);
				} else {
					grad -= 2.0 * this.lectureTradeOff * (obs - pred);
				}
			} else if (resource == 2) {
				pred = getDiscussionPrediction(student, attempt, material);
				if (this.binarizedDiscussion) {
					grad -= 2.0 * this.discussionTradeOff * (obs - pred) * pred * (1.0 - pred);
				} else {
					grad -= 2.0 * this.discussionTradeOff * (obs - pred);
				}
			}
		}
		grad += 2.0 * this.lambda_bias * this.bias_s[student];
		return grad;
	}

	public Double gradBias_t(Integer student, Integer attempt, Integer material, Double obs, Integer resource) throws Exception {
		/**
		 * compute the gradient of loss w.r.t a specific bias_t
		 */

		Double grad = 0.0;
		if (obs != null) {
			Double pred;
			if (resource == 0) {
				pred = getQuestionPrediction(student, attempt, material);
				if (this.binarizedQuestion) {
					grad -= 2.0 * (obs - pred) * pred * (1.0 - pred);
				} else {
					grad -= 2.0 * (obs - pred);
				}
			} else if (resource == 1) {
				pred = getLecturePrediction(student, attempt, material);
				if (this.binarizedLecture) {
					grad -= 2.0 * this.lectureTradeOff * (obs - pred) * pred * (1.0 - pred);
				} else {
					grad -= 2.0 * this.lectureTradeOff * (obs - pred);
				}
			} else if (resource == 2) {
				pred = getDiscussionPrediction(student, attempt, material);
				if (this.binarizedDiscussion) {
					grad -= 2.0 * this.discussionTradeOff * (obs - pred) * pred * (1.0 - pred);
				} else {
					grad -= 2.0 * this.discussionTradeOff * (obs - pred);
				}
			}
		}
		grad += 2.0 * this.lambda_bias * this.bias_t[attempt];
		return grad;
	}

	public Double gradBias_q(Integer student, Integer attempt, Integer question, Double obs) throws Exception {
		/**
		 * compute the gradient of loss w.r.t a specific bias_q
		 */

		Double grad = 0.0;
		if (obs != null) {
			Double pred = getQuestionPrediction(student, attempt, question);
			if (this.binarizedQuestion) {
				grad -= 2.0 * (obs - pred) * pred * (1.0 - pred);
			} else {
				grad -= 2.0 * (obs - pred);
			}
		}

		grad += 2.0 * this.lambda_bias * this.bias_q[question];
		return grad;
	}

	public Double gradBias_l(Integer student, Integer attempt, Integer lecture, Double obs) throws Exception {
		/**
		 * compute the gradient of loss w.r.t a specific bias_l
		 */

		Double grad = 0.0;
		if (obs != null) {
			Double pred = getLecturePrediction(student, attempt, lecture);
			if (this.binarizedLecture) {
				grad -= 2.0 * (obs - pred) * pred * (1.0 - pred);
			} else {
				grad -= 2.0 * (obs - pred);
			}
		}

		grad += 2.0 * this.lambda_bias * this.bias_l[lecture];
		return grad;
	}

	public Double gradBias_d(Integer student, Integer attempt, Integer discussion, Double obs) throws Exception {
		/**
		 * compute the gradient of loss w.r.t a specific bias_d
		 */

		Double grad = 0.0;
		if (obs != null) {
			Double pred = getDiscussionPrediction(student, attempt, discussion);
			if (this.binarizedDiscussion) {
				grad -= 2.0 * (obs - pred) * pred * (1.0 - pred);
			} else {
				grad -= 2.0 * (obs - pred);
			}
		}

		grad += 2.0 * this.lambda_bias * this.bias_d[discussion];
		return grad;
	}

	public void optimzeSGD(Integer student, Integer attempt, Integer material, Double obs, Integer resource)
			throws Exception {
		/**
		 * train the T and Q with stochastic gradient descent
		 */

		// train the bias(es)
		if (resource == 0) {
			this.bias_q[material] -= this.lr * gradBias_q(student, attempt, material, obs);
		} else if (resource == 1) {
			this.bias_l[material] -= this.lr * gradBias_l(student, attempt, material, obs);
		} else if (resource == 2) {
			this.bias_d[material] -= this.lr * gradBias_d(student, attempt, material, obs);
		}

		this.bias_s[student] -= this.lr * gradBias_s(student, attempt, material, obs, resource);

		if (this.useBiasT) {
			this.bias_t[attempt] -= this.lr * gradBias_t(student, attempt, material, obs, resource);
		}

		// optimize T
		Double[][] grad_t = gradT_ij(student, attempt, material, obs, resource);
		// TODO
		setSlice(this.T, attempt, matSub(slice(this.T, attempt), scalarMulMat(this.lr, grad_t)));

		// optimize S
		Double[] grad_s = gradS_k(student, attempt, material, obs, resource);
		this.S[student] = subVectors(this.S[student], scalarMulVector(this.lr, grad_s));
		for (int i = 0; i < this.S[student].length; i++) {
			if (this.S[student][i] < 0.0) {
				this.S[student][i] = 0.0;
			}
		}

		if (this.lambda_s == 0) {
			Double sumVal = sum(this.S[student]);
			if (sumVal != 0.0) {
				this.S[student] = vectorDivScalar(this.S[student], sumVal);
			}
		}

		// update Q if current test attempt is still small, otherwise keep Q the same
		// if self.current_test_attempt < (self.num_attempts / 2):
		if (resource == 0) { // optimize Q
			Double[] grad_q = gradQ_k(student, attempt, material, obs);
			setSlice(this.Q, material, subVectors(slice(this.Q, material), scalarMulVector(this.lr, grad_q)));
			for (int i = 0; i < this.Q.length; i++) {
				if (this.Q[i][material] < 0.0) {
					this.Q[i][material] = 0.0;
				}
			}

			if (this.lambda_q == 0.0) {
				Double sumVal = sum(slice(this.Q, material));
				if (sumVal != 0) {
					setSlice(this.Q, material, vectorDivScalar(slice(this.Q, material), sumVal));
				}
			}
		} else if (resource == 1) { // optimize L
			Double[] grad_l = gradL_k(student, attempt, material, obs);
			setSlice(this.L, material, subVectors(slice(this.L, material), scalarMulVector(this.lr, grad_l)));
			for (int i = 0; i < this.L.length; i++) {
				if (this.L[i][material] < 0.0) {
					this.L[i][material] = 0.0;
				}
			}

			if (this.lambda_l == 0.0) {
				Double sumVal = sum(slice(this.L, material));
				if (sumVal != 0) {
					setSlice(this.L, material, vectorDivScalar(slice(this.L, material), sumVal));
				}
			}
		} else if (resource == 2) { // optimize D
			Double[] grad_d = gradL_k(student, attempt, material, obs);
			setSlice(this.D, material, subVectors(slice(this.D, material), scalarMulVector(this.lr, grad_d)));
			for (int i = 0; i < this.D.length; i++) {
				if (this.D[i][material] < 0.0) {
					this.D[i][material] = 0.0;
				}
			}

			if (this.lambda_d == 0.0) {
				Double sumVal = sum(slice(this.D, material));
				if (sumVal != 0) {
					setSlice(this.D, material, vectorDivScalar(slice(this.D, material), sumVal));
				}
			}
		}
	}

	public ArrayList<Double> training() throws Exception {
		/**
		 * minimize the loss until converged or reach the maximum iterations with
		 * stochastic gradient descent
		 */

		Double[] losses = this.getLoss();
		Double loss = losses[0];
		Double q_count = losses[1];
		Double q_rmse = losses[2];
		Double l_rmse = losses[3];
		Double d_rmse = losses[4];
		Double penality = losses[5];
		Double reg_features = losses[6];
		Double reg_bias = losses[7];

		ArrayList<Double> loss_list = new ArrayList<>();
		loss_list.add(loss);

		ArrayList<ArrayList<Double>> train_perf = new ArrayList<>();
		Boolean converge = false;
		Integer iter_num = 0;
		Integer min_iter = 10;
		Double[][] best_S = null, best_Q = null, best_L = null, best_D = null;
		Double[][][] best_T = null;
		Double[] bestBias_s = null, bestBias_t = null, bestBias_q = null, bestBias_l = null, bestBias_d = null;

		while (!converge) {
			Collections.shuffle(this.trainSet);
			Collections.shuffle(this.trainDataMarkovian);
			best_S = copy(this.S);
			best_T = copy(this.T);
			best_Q = copy(this.Q);
			best_L = copy(this.L);
			best_D = copy(this.D);
			bestBias_s = copy(this.bias_s);
			bestBias_t = copy(this.bias_t);
			bestBias_q = copy(this.bias_q);
			bestBias_l = copy(this.bias_l);
			bestBias_d = copy(this.bias_d);

			for (ArrayList<Number> innerList : this.trainSet) {
				Integer student = (Integer) innerList.get(0);
				Integer attempt = (Integer) innerList.get(1);
				Integer index = (Integer) innerList.get(2);
				Double obs = (Double) innerList.get(3);
				Integer resource = (Integer) innerList.get(4);

				this.optimzeSGD(student, attempt, index, obs, resource);
			}

			for (ArrayList<Number> innerList : this.trainDataMarkovian) {
				Integer student = (Integer) innerList.get(0);
				Integer attempt = (Integer) innerList.get(1);
				Integer index = (Integer) innerList.get(2);
				Integer resource = (Integer) innerList.get(3);

				this.optimzeSGD(student, attempt, index, null, resource);
			}

			losses = this.getLoss();
			loss = losses[0];
			q_count = losses[1];
			q_rmse = losses[2];
			l_rmse = losses[3];
			d_rmse = losses[4];
			penality = losses[5];
			reg_features = losses[6];
			reg_bias = losses[7];

			ArrayList<Double> temp = new ArrayList<>();
			temp.add(q_count);
			temp.add(q_rmse);
			train_perf.add(temp);

			if (iter_num == this.iter) {
				loss_list.add(loss);
				converge = true;
			} else if (iter_num >= min_iter
					&& loss >= mean(loss_list.subList(loss_list.size() - 5, loss_list.size()))) {
				converge = true;
			} else if (loss.isNaN()) {
				this.lr *= 0.1;
				throw new Exception("NAN");
			} else if (loss > loss_list.get(loss_list.size() - 1)) {
				loss_list.add(loss);
				this.lr *= 0.5;
				iter_num++;
			} else {
				loss_list.add(loss);
				iter_num++;
			}
		}

		this.S = best_S;
		this.T = best_T;
		this.Q = best_Q;
		this.L = best_L;
		this.D = best_D;
		this.bias_s = bestBias_s;
		this.bias_t = bestBias_t;
		this.bias_q = bestBias_q;
		this.bias_l = bestBias_l;
		this.bias_d = bestBias_d;

		return train_perf.get(train_perf.size() - 1);
	}

	public LinkedHashMap<String, Double> testing(ArrayList<ArrayList<Number>> test_data) throws Exception {
		/**
		 * @return: performance metrics mean squared error, RMSE, and mean absolute
		 *          error
		 */

		ArrayList<Double> curr_pred_list = new ArrayList<>();
		ArrayList<Double> curr_obs_list = new ArrayList<>();

		for (ArrayList<Number> innerList : test_data) {
			Integer student = (Integer) innerList.get(0);
			Integer attempt = (Integer) innerList.get(1);
			Integer question = (Integer) innerList.get(2);
			Double obs = (Double) innerList.get(3);
			Integer resource = (Integer) innerList.get(4);

			Double pred;
			if (resource == 0) {
				curr_obs_list.add(obs);
				pred = this.getQuestionPrediction(student, attempt, question);
				curr_pred_list.add(pred);
				Double square_error = Math.pow((obs - pred), 2);
				this.testObsList.add(obs);
				this.testPredList.add(pred);
			}
		}

		return this.eval(curr_obs_list, curr_pred_list);
	}

	public LinkedHashMap<String, Double> eval(ArrayList<Double> obsList, ArrayList<Double> predList) {
		/**
		 * evaluate the prediction performance
		 */

		assert predList.size() == obsList.size();

		double count = obsList.size();
		LinkedHashMap<String, Double> prefMap = new LinkedHashMap<>();
		if (predList.size() == 0) {
			return prefMap;
		} else {
			prefMap.put("count", count);
		}

		for (String metric : this.metrics) {
			if (metric.equals("rmse")) {
				Double rmse = rootMeanSquaredError(obsList, predList);
				prefMap.put(metric, rmse);
			} else if (metric.equals("mae")) {
				Double mae = meanAbsoluteError(obsList, predList);
				prefMap.put(metric, mae);
			}

			/*
			 * TODO: need to implement ROC AUC algorithm else if(metric.equals("auc")){
			 * if(sum(obsList) == count){ prefMap.put(metric, null); } else{ Double auc =
			 * rocAucScore(obsList, predList); prefMap.put(metric, auc); } }
			 */
		}

		return prefMap;
	}

	public void generateNextItems() throws Exception {
		/**
		 * decompose the current outcome observations into knowledge and concept-question
		 * association matrix, then generate next question which is relative difficult
		 * than previous question here we need to keep updating and tracking all
		 * students' current_questions
		 * 
		 * @param threshold is a hyper parameter we need to tune
		 * 
		 *                  TODO: not just difficulty but also use the association of
		 *                  questions
		 */
		Integer hit_count = 0;
		Integer miss_count = 0;
		for (Integer student : this.testUsers.keySet()) {
			// if the test user never done so many attempts, then continue to next test user
			if (!this.testUserRecords.get(student).containsKey(this.startAttempt)) {
				continue;
			}

			// compute the updated proximity score of questions for a specific student
			Integer curr_ques = this.currentQuestion.get(student);
			Double curr_score = this.currentScore.get(student);

			// each question has a proximity value, we need to update it after each
			// interaction
			ArrayList<Double> updatedList = new ArrayList<>();
			for (int question = 0; question < this.numQuestions; question++) {
				// estimate the knowledge gap before the current_test_attempt
				Double est_score = this.getQuestionPrediction(student, this.startAttempt - 1, question);
				Double[] knowledge = dot(this.S[student], slice(this.T, this.startAttempt - 1));
				knowledge = scalarPlusVector(knowledge, this.bias_s[student]);
				Double gap = cosine(knowledge, slice(this.Q, question));
				Double proximity = (1.0 - gap) * (1.0 - est_score);
				// we use the true score instead of estimated score if we observe it
				if (curr_ques == question) {
					Double trueQuestionEstScore = est_score;
					proximity = (1.0 - gap) * (1.0 - curr_score);
				}

				// update the proximity score for each question
				updatedList.add(proximity);
			}
			ArrayList<Double> sortedList = new ArrayList<>();
			sortedList.addAll(updatedList);
			Collections.sort(sortedList, Collections.reverseOrder());

			// option 1:
			ArrayList<Integer> candidates = new ArrayList<>();
			ArrayList<ArrayList<Number>> canidates_and_proximity = new ArrayList<>();

			// mean_score_current_question = np.mean(self.question_score_dict[curr_ques])
			for (Double prox : sortedList) {
				if (candidates.size() < this.top_k) { // try to get k candidates in maximum
					Integer next_question = updatedList.indexOf(prox);
					candidates.add(next_question);
					ArrayList<Number> temp = new ArrayList<>();
					temp.add(next_question);
					temp.add(prox);
					canidates_and_proximity.add(temp);
				}
			}

			for (Integer cand : candidates) {
				if (!this.recMap.containsKey(cand)) {
					this.recMap.put(cand, 0);
				}

				Integer temp = this.recMap.get(cand) + 1;
				this.recMap.replace(cand, temp);
			}

			// compute the IPW-nDCH for each round of recommendation
			Integer trueNextQuestion = this.testUserRecords.get(student).get(this.startAttempt);
			if (candidates.contains(trueNextQuestion)) {
				// if the recommendation was also selected by the user in logged system.
				hit_count++;
				// compute ipw-ndch
				Double ndch = 1.0 / log2(candidates.indexOf(trueNextQuestion) + 1.0 + 1.0);
				Integer total_count = 0;
				for (Integer question : this.nextQuestionsMap.get(curr_ques).keySet()) {
					total_count += this.nextQuestionsMap.get(curr_ques).get(question);
				}
				
				Double propensity = this.nextQuestionsMap.get(curr_ques).get(trueNextQuestion).doubleValue() / total_count.doubleValue();
				Double ipw_ndch = ndch / propensity;
				this.recommendation.get(student).add(ipw_ndch);
			} else {
				// if the recommendation was not selected by the user in logged system.
				miss_count++;
				this.recommendation.get(student).add(0.0);
			}

			// append curr_ques into historical records
			if (!this.historicalRecords.get(student).containsKey(curr_ques)) {
				this.historicalRecords.get(student).put(curr_ques, new ArrayList<>());
			}

			this.historicalRecords.get(student).get(curr_ques).add(curr_score);
			// update the current question
			this.currentQuestion.replace(student, trueNextQuestion);
			this.currentScore.replace(student, (Double) this.userData.get(student).get(this.startAttempt).get(3));
		}
		
		this.total_hit += hit_count;
		this.total_miss += miss_count;
	}
	
	public Double[] getLoss(LinkedHashMap<String, Double> trainDataMap, LinkedHashMap<String, Boolean> trainDataMarkovMap) throws Exception {
		/**
		 * override the function in super class compute the loss, which is RMSE of
		 * observed records + regularization + penalty of temporal non-smoothness
		 * 
		 * @return: loss
		 */

		Double loss = 0.0;
		Double square_loss = 0.0;
		Double reg_bias = 0.0;
		Double square_loss_q = 0.0;
		Double q_count = 0.0;
		
		for(String key : trainDataMap.keySet()) {
			String[] nums = key.split(",");
			Integer student = Integer.parseInt(nums[0]);
			Integer time_index = Integer.parseInt(nums[1]);
			Integer question = Integer.parseInt(nums[2]);
			
			Double obs = trainDataMap.get(key);
			Integer confd = this.confidenceMap.get(student).get(question).get(time_index);
			Double pred = this.getQuestionPrediction(student, time_index, question);
			square_loss_q += confd * Math.pow(obs-pred, 2);
			q_count++;
		}
		
		square_loss = square_loss_q;
		
		// regularzation of student data
		Double reg_S = norm(this.S);
		Double reg_T = norm(this.T);
		Double reg_Q = norm(this.Q);
		Double reg_loss = this.lambda_s * reg_S + this.lambda_q * reg_Q + this.lambda_t * reg_T;
		
		Double q_rmse = (q_count != 0) ? Math.sqrt(square_loss_q / q_count) : 0;
		
		if (this.lambda_bias != null) {
			if (this.useBiasT) {
				reg_bias = this.lambda_bias * (norm(this.bias_s) + norm(this.bias_t) + norm(this.bias_q));
			} else {
				reg_bias = this.lambda_bias * (norm(this.bias_s) + norm(this.bias_q));
			}
		}
		
		Double penalty;
		if(this.penality_weight != 0 && trainDataMarkovMap != null) {
			penalty = this.getPenalty(trainDataMap, trainDataMarkovMap);
		}
		else {
			penalty = 0.0;
		}

		loss = square_loss + reg_loss + reg_bias + penalty;
		Double[] losses = { loss, q_count, q_rmse, penalty, reg_loss, reg_bias };
		return losses;
	}
	
	public Double getPenalty(LinkedHashMap<String, Double> trainDataMap, LinkedHashMap<String, Boolean> trainDataMarkovMap) throws Exception {
		/**
		 * compute the penalty on the observations, we want all attempts before the obs
		 * has smaller score, and the score after obs should be greater. we use sigmoid
		 * to set the penalty between 0 and 1 if knowledge at current attempt >> prev
		 * attempt, then diff is large, that mean sigmoid(diff) is large and close to
		 * 1., so penalty is a very small negative number since we aim to minimize the
		 * objective = loss + penalty, the smaller penalty is better
		 */

		Double penality = 0.0;
		for(String key : trainDataMap.keySet()) {
			String[] nums = key.split(",");
			Integer student = Integer.parseInt(nums[0]);
			Integer time_index = Integer.parseInt(nums[1]);
			Integer index = Integer.parseInt(nums[2]);
			
			Double obs = trainDataMap.get(key);

			if (time_index >= 1) {
				Double[][] gap = matSub(slice(this.T, time_index), slice(this.T, time_index - 1));
				Double[] knowledge_gap = dot(this.S[student], gap);
				Double penality_val;
				if (this.exactPenalty) {
					for (int i = 0; i < knowledge_gap.length; i++) {
						if (knowledge_gap[i] > 0.0) {
							knowledge_gap[i] = 0.0;
						}
					}

					penality_val = -1.0 * dot(knowledge_gap, slice(this.Q, index));
				} else {
					Double diff = dot(knowledge_gap, slice(this.Q, index));
					penality_val = -1.0 * Math.log(sigmoid(diff, false));
				}
				penality += this.penality_weight * penality_val;
			}
		}

		for(String key : trainDataMarkovMap.keySet()) {
			String[] nums = key.split(",");
			Integer student = Integer.parseInt(nums[0]);
			Integer time_index = Integer.parseInt(nums[1]);
			Integer index = Integer.parseInt(nums[2]);

			if (time_index >= 1) {
				Double[][] gap = matSub(slice(this.T, time_index), slice(this.T, time_index - 1));
				Double[] knowledge_gap = dot(this.S[student], gap);
				Double penality_val;
				if (this.exactPenalty) {
					for (int i = 0; i < knowledge_gap.length; i++) {
						if (knowledge_gap[i] > 0.0) {
							knowledge_gap[i] = 0.0;
						}
					}

					penality_val = -1.0 * dot(knowledge_gap, slice(this.Q, index));
				} else {
					Double diff = dot(knowledge_gap, slice(this.Q, index));
					penality_val = -1.0 * Math.log(sigmoid(diff, false));
				}
				penality += this.penality_weight * penality_val;
			}
		}

		return penality;
	}
	
	public ArrayList<Double> fastTraining() throws Exception {
		/**
		 * minimize the loss until converged or reach the maximum iterations
         * with stochastic gradient descent
		 */
		
		 ArrayList<ArrayList<Double>> train_perf = new ArrayList<>();
		 Boolean converage = false;
		 Integer iter_num = 0;
		 Integer min_iter = 10;
		 
		 Double[][] best_S = copy(this.S);
		 Double[][][] best_T = copy(this.T);
		 Double[][] best_Q = copy(this.Q);
		 Double[] best_bias_s = copy(this.bias_s);
		 Double[] best_bias_t = copy(this.bias_t);
		 Double[] best_bias_q = copy(this.bias_t);
		 
		 LinkedHashMap<String, Double> trainDataMap = new LinkedHashMap<>();
		 LinkedHashMap<String, Boolean> trainDataMarkovMap = new LinkedHashMap<>();
		 this.confidenceMap = new LinkedHashMap<>();
		 
		 Integer timeWindow;
		 if(this.dataStr.equals("morf")) {
			 timeWindow = 3;
		 }
		 else {
			 timeWindow = 1;
		 }
		 
		 for(ArrayList<Number> innerList : this.trainSet) {
			 Integer student = (Integer) innerList.get(0);
			 Integer time_index = (Integer) innerList.get(1);
			 Integer question = (Integer) innerList.get(2);
			 Double score = (Double) innerList.get(3);
			 
			 if((-1*timeWindow) <= (this.startAttempt - time_index) && (this.startAttempt - time_index) <= timeWindow) {
				 String key = student.toString() + "," + time_index.toString() + "," + question.toString();
				 trainDataMap.put(key, score);
				 
				 if(!this.confidenceMap.containsKey(student)) {
					 this.confidenceMap.put(student, new LinkedHashMap<>());
				 }
				 if(!this.confidenceMap.get(student).containsKey(question)) {
					 this.confidenceMap.get(student).put(question, new LinkedHashMap<>());
				 }
				 if(!this.confidenceMap.get(student).get(question).containsKey(time_index)) {
					 Integer num_trials = this.confidenceMap.get(student).get(question).size();
					 if(this.dataStr.equals("mastery_grids")) {
						 this.confidenceMap.get(student).get(question).put(time_index, 1);
					 }
					 else {
						 this.confidenceMap.get(student).get(question).put(time_index, num_trials+1);
					 }
				 }
			 }
		 }
		 
		 for(String key : trainDataMap.keySet()) {
			 String[] nums = key.split(",");
			 Integer student = Integer.parseInt(nums[0]);
			 Integer time_index = Integer.parseInt(nums[1]);
			 Integer question = Integer.parseInt(nums[2]);
			 
			 Integer upper_steps = Math.min(this.numAttempts, time_index+this.markovian+1);
			 for(Integer i=time_index+1; i<upper_steps; i++) {
				 String newKey = student.toString() + "," + i.toString() + "," + question.toString();
				 if(!trainDataMap.containsKey(newKey)) {
					 if(!trainDataMarkovMap.containsKey(newKey)) {
						 trainDataMarkovMap.put(newKey, true);
					 }
				 }
			 }
			 
			 Integer lower_steps = Math.max(0, time_index - this.markovian);
			 for(Integer i=lower_steps; i<time_index; i++) {
				 String newKey = student.toString() + "," + i.toString() + "," + question.toString();
				 if(!trainDataMap.containsKey(newKey)) {
					 if(!trainDataMarkovMap.containsKey(newKey)) {
						 trainDataMarkovMap.put(newKey, true);
					 }
				 }
			 }
		 }
		 
		 ArrayList<String> trainData = new ArrayList<>();
		 trainData.addAll(trainDataMap.keySet());
		 
		 ArrayList<String> trainDataMarkov = new ArrayList<>();
		 trainDataMarkov.addAll(trainDataMarkovMap.keySet());
		 
		 Double[] losses = this.getLoss(trainDataMap, trainDataMarkovMap);
		 Double loss = losses[0];
		 Double count = losses[1];
		 Double rmse = losses[2];
		 Double penalty = losses[3];
		 Double reg_loss = losses[4];
		 Double reg_bias = losses[5];
		 
		 ArrayList<Double> loss_list = new ArrayList<>();
		 loss_list.add(loss);
		 
		 while(!converage) {
			 Collections.shuffle(trainData);
			 Collections.shuffle(trainDataMarkov);
			 best_S = copy(this.S);
			 best_T = copy(this.T);
			 best_Q = copy(this.Q);
			 best_bias_s = copy(this.bias_s);
			 best_bias_t = copy(this.bias_t);
			 best_bias_q = copy(this.bias_q);
			 
			 for(String key : trainData) {
				 String[] nums = key.split(",");
				 Integer student = Integer.parseInt(nums[0]);
				 Integer time_index = Integer.parseInt(nums[1]);
				 Integer question = Integer.parseInt(nums[2]);
				 
				 Double obs = trainDataMap.get(key);
				 this.optimzeSGD(student, time_index, question, obs, 0);
			 }
			 for(String key : trainDataMarkov) {
				 String[] nums = key.split(",");
				 Integer student = Integer.parseInt(nums[0]);
				 Integer time_index = Integer.parseInt(nums[1]);
				 Integer question = Integer.parseInt(nums[2]);
				 
				 this.optimzeSGD(student, time_index, question, null, 0);
			 }
			 
			 losses = this.getLoss(trainDataMap, trainDataMarkovMap);
			 loss = losses[0];
			 count = losses[1];
			 rmse = losses[2];
			 penalty = losses[3];
			 reg_loss = losses[4];
			 reg_bias = losses[5];
			 
			 ArrayList<Double> temp = new ArrayList<>();
			 temp.add(count);
			 temp.add(rmse);
			 train_perf.add(temp);
			 
			 if(iter_num == this.iter) {
				 loss_list.add(loss);
				 converage = true;
			 }
			 else if(iter_num >= min_iter && loss >= mean(loss_list.subList(loss_list.size()-5, loss_list.size()))) {
				 converage = true;
			 }
			 else if(loss.isNaN()) {
				 this.lr *= 0.1;
			 }
			 else if(loss > loss_list.get(loss_list.size()-1)) {
				 loss_list.add(loss);
				 this.lr *= 0.5;
				 iter_num++;
			 }
			 else {
				 loss_list.add(loss);
				 iter_num++;
			 }
		 }
		 
		 this.S = best_S;
		 this.T = best_T;
		 this.Q = best_Q;
		 this.bias_s = best_bias_s;
		 this.bias_t = best_bias_t;
		 this.bias_q = best_bias_q;
		 
		 return train_perf.get(train_perf.size()-1);
	}

	private Double[][] randomSample(Integer n, Integer m) {
		Double[][] matrix = new Double[n][m];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				matrix[i][j] = Math.random();
			}
		}

		return matrix;
	}

	private Double[][][] randomSample(Integer n, Integer m, Integer d) {
		Double[][][] tensor = new Double[n][m][d];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				for (int k = 0; k < d; k++) {
					tensor[i][j][k] = Math.random();
				}
			}
		}

		return tensor;
	}

	private Double[][] zeros(Integer n, Integer m) {
		Double[][] matrix = new Double[n][m];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				matrix[i][j] = 0.0;
			}
		}

		return matrix;
	}

	private Double[] zeros(Integer n) {
		Double[] vector = new Double[n];
		for (int i = 0; i < n; i++) {
			vector[i] = 0.0;
		}

		return vector;
	}

	private Double mean(List<Double> l) {
		Double sum = 0.0;
		for (Double d : l) {
			sum += d;
		}

		return sum / l.size();
	}

	private Double[] meanAxis0(ArrayList<ArrayList<Number>> data) {
		Double studentSum = 0.0;
		Double attemptSum = 0.0;
		Double questionSum = 0.0;
		Double obsSum = 0.0;
		Double resourceSum = 0.0;
		for (ArrayList<Number> innerList : data) {
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
		Double[] means = { studentSum / size, attemptSum / size, questionSum / size, obsSum / size,
				resourceSum / size };
		return means;
	}

	private Double rootMeanSquaredError(ArrayList<Double> y_truth, ArrayList<Double> y_pred) {
		Double error = 0.0;
		for (int i = 0; i < y_truth.size(); i++) {
			error += Math.pow(y_truth.get(i) - y_pred.get(i), 2);
		}

		return Math.sqrt(error / y_truth.size());
	}

	private Double meanAbsoluteError(ArrayList<Double> y_truth, ArrayList<Double> y_pred) {
		Double error = 0.0;
		for (int i = 0; i < y_truth.size(); i++) {
			error += Math.abs(y_truth.get(i) - y_pred.get(i));
		}

		return error / y_truth.size();
	}

	/*
	 * TODO private Double rocAucScore(ArrayList<Double> y_truth, ArrayList<Double>
	 * y_pred){
	 * 
	 * }
	 */

	private Double[][] slice(Double[][][] tensor, Integer slice) {
		Double[][] matrix = new Double[tensor.length][tensor[0][slice].length];
		for (int i = 0; i < tensor.length; i++) {
			for (int j = 0; j < tensor[i][slice].length; j++) {
				matrix[i][j] = tensor[i][slice][j];
			}
		}

		return matrix;
	}

	private Double[] slice(Double[][] matrix, Integer slice) {
		Double[] vector = new Double[matrix.length];
		for (int i = 0; i < matrix.length; i++) {
			vector[i] = matrix[i][slice];
		}

		return vector;
	}

	private void setSlice(Double[][][] tensor, Integer slice, Double[][] newMatrix) throws Exception {
		for (int i = 0; i < newMatrix.length; i++) {
			for (int j = 0; j < newMatrix[i].length; j++) {
				if (newMatrix[i][j].isNaN()) {
					throw new Exception();
				}
				tensor[i][slice][j] = newMatrix[i][j];
			}
		}
	}

	private void setSlice(Double[][] matrix, Integer slice, Double[] newVector) throws Exception {
		for (int i = 0; i < newVector.length; i++) {
			if (newVector[i].isNaN()) {
				throw new Exception();
			}
			matrix[i][slice] = newVector[i];
		}
	}

	private Double[] dot(Double[] vector, Double[][] matrix) {
		Double[] result = new Double[matrix[0].length];

		for (int i = 0; i < matrix[0].length; i++) {
			result[i] = 0.0;
			for (int j = 0; j < vector.length; j++) {
				result[i] += vector[j] * matrix[j][i];
			}
		}

		return result;
	}

	private Double dot(Double[] v1, Double[] v2) {
		Double result = 0.0;
		for (int i = 0; i < v1.length; i++) {
			result += v1[i] * v2[i];
		}

		return result;
	}

	private Double[] dot(Double[][] matrix, Double[] vector) {
		Double[] result = new Double[matrix.length];
		for (int i = 0; i < matrix.length; i++) {
			result[i] = 0.0;
			for (int j = 0; j < matrix[i].length; j++) {
				result[i] += matrix[i][j] * vector[j];
			}
		}

		return result;
	}

	private Double[][] outer(Double[] v1, Double[] v2) {
		Double[][] result = new Double[v1.length][v2.length];
		for (int i = 0; i < v1.length; i++) {
			for (int j = 0; j < v2.length; j++) {
				result[i][j] = v1[i] * v2[j];
			}
		}

		return result;
	}

	private Double sigmoid(Double x, Boolean derivative) throws Exception {
		/**
		 * compute the sigmoid function 1 / (1 + exp(-x))
		 * 
		 * @param x:          input of sigmoid function
		 * @param derivative: boolean value, if True compute the derivative of sigmoid
		 *                    function instead
		 * @return:
		 */

		Double sigm;
		if (x > 100.0 || x == Double.POSITIVE_INFINITY) {
			sigm = 1.0;
		} else if (x < -100.0 || x == Double.NEGATIVE_INFINITY) {
			sigm = 0.0;
		} else {
			sigm = 1.0 / (1.0 + Math.exp(-1.0 * x));
		}
		
		if(sigm.isNaN()) {
			throw new Exception();
		}

		if (derivative) {
			return sigm * (1.0 - sigm);
		}

		return sigm;
	}

	private Double norm(Double[][] matrix) {
		Double result = 0.0;
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[i].length; j++) {
				result += Math.pow(matrix[i][j], 2);
			}
		}

		return result;
	}

	private Double norm(Double[][][] tensor) {
		Double result = 0.0;
		for (int i = 0; i < tensor.length; i++) {
			for (int j = 0; j < tensor[i].length; j++) {
				for (int k = 0; k < tensor[i][j].length; k++) {
					result += tensor[i][j][k];
				}
			}
		}

		return result;
	}

	private Double norm(Double[] vector) {
		Double result = 0.0;
		for (int i = 0; i < vector.length; i++) {
			result += Math.pow(vector[i], 2);
		}

		return result;
	}

	private Double normL2(Double[] vector) {
		Double result = 0.0;
		for (int i = 0; i < vector.length; i++) {
			result += Math.pow(vector[i], 2);
		}

		return Math.sqrt(result);
	}

	private Double[][] matSub(Double[][] m1, Double[][] m2) {
		Double[][] result = new Double[m1.length][m1[0].length];
		for (int i = 0; i < m1.length; i++) {
			for (int j = 0; j < m1[i].length; j++) {
				result[i][j] = m1[i][j] - m2[i][j];
			}
		}

		return result;
	}

	private Double[][] matAdd(Double[][] m1, Double[][] m2) {
		Double[][] result = new Double[m1.length][m1[0].length];
		for (int i = 0; i < m1.length; i++) {
			for (int j = 0; j < m1[i].length; j++) {
				result[i][j] = m1[i][j] + m2[i][j];
			}
		}

		return result;
	}

	private Double[][] scalarMulMat(Double s, Double[][] mat) {
		Double[][] result = new Double[mat.length][mat[0].length];
		for (int i = 0; i < mat.length; i++) {
			for (int j = 0; j < mat[i].length; j++) {
				result[i][j] = s * mat[i][j];
			}
		}

		return result;
	}

	private Double[] scalarMulVector(Double s, Double[] vector) {
		Double[] result = new Double[vector.length];
		for (int i = 0; i < vector.length; i++) {
			result[i] = vector[i] * s;
		}

		return result;
	}

	private Double[][] matPlusScalar(Double[][] mat, Double s) {
		Double[][] result = new Double[mat.length][mat[0].length];
		for (int i = 0; i < mat.length; i++) {
			for (int j = 0; j < mat[0].length; j++) {
				result[i][j] = mat[i][j] + s;
			}
		}

		return result;
	}

	private Double[] addVectors(Double[] v1, Double[] v2) {
		Double[] result = new Double[v1.length];
		for (int i = 0; i < v1.length; i++) {
			result[i] = v1[i] + v2[i];
		}

		return result;
	}

	private Double[] subVectors(Double[] v1, Double[] v2) {
		Double[] result = new Double[v1.length];
		for (int i = 0; i < v1.length; i++) {
			result[i] = v1[i] - v2[i];
		}

		return result;
	}

	private Double[] vectorDivScalar(Double[] v, Double s) {
		Double[] r = new Double[v.length];
		for (int i = 0; i < v.length; i++) {
			r[i] = v[i] / s;
		}

		return r;
	}

	private Double[] scalarPlusVector(Double[] v, Double s) {
		Double[] r = new Double[v.length];
		for (int i = 0; i < v.length; i++) {
			r[i] = v[i] + s;
		}

		return r;
	}

	private Double sum(Double[] v) {
		Double sum = 0.0;
		for (int i = 0; i < v.length; i++) {
			sum += v[i];
		}
		return sum;
	}

	private Double sum(ArrayList<Double> v) {
		Double sum = 0.0;
		for (int i = 0; i < v.size(); i++) {
			sum += v.get(i);
		}
		return sum;
	}

	private Double[][] copy(Double[][] m) throws Exception {
		Double[][] r = new Double[m.length][m[0].length];
		for (int i = 0; i < m.length; i++) {
			for (int j = 0; j < m[i].length; j++) {
				if(m[i][j].isNaN()) {
					throw new Exception();
				}
				r[i][j] = m[i][j];
			}
		}

		return r;
	}

	private Double[][][] copy(Double[][][] t) throws Exception {
		Double[][][] r = new Double[t.length][t[0].length][t[0][0].length];
		for (int i = 0; i < t.length; i++) {
			for (int j = 0; j < t[i].length; j++) {
				for (int k = 0; k < t[i][j].length; k++) {
					if(t[i][j][k].isNaN()) {
						throw new Exception();
					}
					r[i][j][k] = t[i][j][k];
				}
			}
		}
		return r;
	}

	private Double[] copy(Double[] v) {
		Double[] r = new Double[v.length];
		for (int i = 0; i < v.length; i++) {
			r[i] = v[i];
		}
		return r;
	}

	private Double cosine(Double[] u, Double[] v) {
		Double dot_uv = dot(u, v);
		return 1 - (dot_uv / (normL2(u) * normL2(v)));
	}

	private Double log2(Double x) {
		return Math.log(x) / Math.log(2);
	}

	private void printMatrix(Double[][] matrix) {
		for (Double[] arr : matrix) {
			for (Double d : arr) {
				System.out.print(d + " ");
			}
			System.out.println();
		}
	}

	private void printVector(Double[] vector) {
		for (Double d : vector) {
			System.out.print(d + " ");
		}
		System.out.println();
	}
}