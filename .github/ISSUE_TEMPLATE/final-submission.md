## Team name: 
Xyzzy

## Team members: 
Yudai Satoh

## Project Description: 
  In this project, I tried to reduce the execution time per step because I believe it is important to search for optimal parameters as many times as possible.To achieve this, I used SPSA as an optimizer and reduced shots by stochastically selecting measuring terms of the Hamiltonian. In addition, I implemented ZNE to obtain smaller expected values. However, the value from ZNE has instability. Therefore, I calculated outliers of the costs during optimization and used it to decide the acceptable range of ZNE results. 

## Presentation:
https://1drv.ms/p/s!AmKF5Bc81V_y71k9BLnVMPTqp1qP?e=zFPQAX

## Source code: 
https://github.com/lambyudG/QAGC_Submission/blob/main/problem/answer.py
