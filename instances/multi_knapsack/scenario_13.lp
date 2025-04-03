Minimize
 obj: - 18 x_0_0 - 16 x_0_1 - 16 x_1_0 - 17 x_1_1 - 15 x_2_0 - 15 x_2_1 
 - 19 x_3_0 - 16 x_3_1 

Subject To 
 ohc_item_0: + 1 x_0_0 + 1 x_0_1 + 1 x_0_2  = 1.0
 ohc_item_1: + 1 x_1_0 + 1 x_1_1 + 1 x_1_2  = 1.0
 ohc_item_2: + 1 x_2_0 + 1 x_2_1 + 1 x_2_2  = 1.0
 ohc_item_3: + 1 x_3_0 + 1 x_3_1 + 1 x_3_2  = 1.0
 knapsack_0: + 7 x_0_0 + 1 x_1_0 + 5 x_2_0 + 7 x_3_0  <= 10.0
 knapsack_1: + 7 x_0_1 + 1 x_1_1 + 5 x_2_1 + 7 x_3_1  <= 10.0

Bounds

Binary
 x_0_0 x_0_1 x_1_0 x_1_1 x_2_0 x_2_1 x_3_0 x_3_1 x_0_2 x_1_2 x_2_2 x_3_2
General

End