# IT Support Cost Optimization Using Data Analysis
**Author:** Manus AI

## Executive Summary
This project analyzes 100,000 synthetic IT support tickets (scalable to 2 million/year) to identify cost drivers, improve customer satisfaction, and optimize engineer allocation. The analysis provides actionable recommendations to reduce the total annual support cost of $39.8M while maintaining high service quality.

## 1. Project Goals
1. **Reduce total support cost by 20%** (Target: ~$8M savings).
2. **Improve customer satisfaction** above 4.2.
3. **Reduce unnecessary escalations**.
4. **Optimize engineer allocation** based on ticket volume and resolution time.

## 2. Data Strategy & Scalability
### Missing Data Strategy
The dataset contained ~20% missing values across key fields. The following imputation strategies were implemented:
- **Issue Category:** Imputed with 'Unknown' rather than the mode. At 20% missingness, mode imputation would artificially skew the distribution. Keeping it as 'Unknown' allows tracking if missing data correlates with higher costs.
- **Priority:** Imputed with the Mode (Medium), representing the standard default ticket state.
- **Resolution Time:** Imputed using the Median grouped by Priority. This accounts for the strong dependency between priority and resolution time while avoiding the influence of extreme outliers.
- **Satisfaction Score:** Imputed with the Median to represent the typical customer experience.

### Scalability Approach
The data generation and cleaning pipelines are built using vectorized Pandas operations. The ML model chosen (Logistic Regression) is highly optimized for fast inference, capable of processing 10,000 predictions in ~0.01 seconds, easily handling the simulated streaming requirement of 50,000 events/second.

## 3. Exploratory Data Analysis (EDA)
Key findings from the visual analysis:

**Cost Drivers:** Software and Hardware issues, particularly at Low and Medium priorities, drive the highest total costs due to sheer volume.
![Cost by Category](https://files.manuscdn.com/user_upload_by_module/session_file/310519663671796482/aygwFtTOBihhrWwA.png)

**Resolution Time:** Critical priority tickets have significantly higher and more variable resolution times, directly impacting costs and satisfaction.
![Resolution Time by Priority](https://files.manuscdn.com/user_upload_by_module/session_file/310519663671796482/AbfgfDZtbjbEAWRG.png)

**Escalations:** Enterprise clients experience the highest average escalation counts, indicating a need for specialized handling.
![Escalation by Client](https://files.manuscdn.com/user_upload_by_module/session_file/310519663671796482/qrUJhHasoppgzDNm.png)

**Satisfaction vs. Resolution Time:** There is a clear inverse relationship between resolution time and customer satisfaction.
![Satisfaction vs Resolution Time](https://files.manuscdn.com/user_upload_by_module/session_file/310519663671796482/HicGqSHUEgvOlejp.png)

## 4. Cost Optimization & Engineer Allocation
### Cost Reduction Strategy
The total annual support cost is calculated at **$39,842,437.90**. To achieve the 20% reduction target ($7.96M), we propose:
1. **Automation of Routine Tickets:** Automating 30% of Low/Medium priority 'Software' and 'Account/Access' tickets. Assuming a 50% cost reduction per automated ticket, this yields an estimated **$3.82M** in savings.
2. **Targeted Training:** Reducing resolution time for Critical 'Network' issues by 20% through specialized engineer training yields an estimated **$228K** in savings.

*Total Projected Savings:* **$4.05M (~10.2% reduction)**. Further process improvements are required to hit the full 20% target.

### Engineer Allocation Recommendation
Based on the total hours required per category, engineers should be allocated as follows to optimize response times:
- **Software:** 26.4%
- **Hardware:** 20.9%
- **Unknown/Uncategorized:** 20.1%
- **Network:** 16.6%
- **Account/Access:** 9.3%
- **Other:** 6.7%

## 5. Escalation Risk Prediction Model
To proactively manage escalations, a lightweight Machine Learning model was deployed.
- **Model Choice:** Logistic Regression. Chosen for its interpretability, low memory footprint, and extremely fast inference speed, making it ideal for real-time streaming environments.
- **Performance:** The model achieved an accuracy of 65% on the synthetic dataset.
- **Inference Benchmark:** Tested on 10,000 records, the total inference time was **0.0122 seconds** (avg 0.000001s per prediction), well within the <5 seconds constraint.

## 6. Actionable Recommendations
1. **Implement Self-Service/Automation:** Focus immediately on Software and Account/Access categories for Low/Medium priority tickets to capture the $3.8M savings opportunity.
2. **Realign Staffing:** Adjust engineer schedules to match the recommended allocation percentages, ensuring adequate coverage for high-volume Software and Hardware issues.
3. **Deploy Escalation Predictor:** Integrate the Logistic Regression model into the ticketing system to flag high-risk tickets at creation, routing them to senior engineers immediately to prevent costly escalations and protect the >4.2 satisfaction goal.
4. **Improve Data Collection:** Investigate why 20% of tickets lack an issue category, as 'Unknown' tickets currently consume 20% of engineer time. Mandatory categorization at ticket creation will improve future modeling and routing.
