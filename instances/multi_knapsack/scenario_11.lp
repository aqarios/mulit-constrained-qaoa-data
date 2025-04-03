Minimize
 obj: - 18 x_0_0 - 18 x_0_1 - 19 x_1_0 - 17 x_1_1 - 17 x_2_0 - 18 x_2_1 

Subject To 
 ohc_item_0: + 1 x_0_0 + 1 x_0_1 + 1 x_0_2  = 1.0
 ohc_item_1: + 1 x_1_0 + 1 x_1_1 + 1 x_1_2  = 1.0
 ohc_item_2: + 1 x_2_0 + 1 x_2_1 + 1 x_2_2  = 1.0
 knapsack_0: + 3 x_0_0 + 6 x_1_0 + 2 x_2_0  <= 8.0
 knapsack_1: + 3 x_0_1 + 6 x_1_1 + 2 x_2_1  <= 11.0

Bounds

Binary
 x_0_0 x_0_1 x_1_0 x_1_1 x_2_0 x_2_1 x_0_2 x_1_2 x_2_2
General

End