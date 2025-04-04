Minimize
 obj: + 232 x_user1_load0_hour0 + 219 x_user1_load0_hour1 
 + 202 x_user1_load0_hour2 + 197 x_user1_load0_hour3 + 184 x_user1_load0_hour4 
 + 169 x_user1_load0_hour5 + 150 x_user1_load0_hour6 + 141 x_user1_load0_hour7 
 + 236 x_user1_load1_hour0 + 214 x_user1_load1_hour1 + 204 x_user1_load1_hour2 
 + 196 x_user1_load1_hour3 + 188 x_user1_load1_hour4 + 166 x_user1_load1_hour5 
 + 148 x_user1_load1_hour6 + 144 x_user1_load1_hour7 + 132 x_user1_load1_hour8 
 + 267 x_user1_load2_hour0 + 258 x_user1_load2_hour1 + 234 x_user1_load2_hour2 
 + 230 x_user1_load2_hour3 + 211 x_user1_load2_hour4 + 198 x_user1_load2_hour5 
 + 176 x_user1_load2_hour6 + 162 x_user1_load2_hour7 + 153 x_user1_load3_hour0 
 + 146 x_user1_load3_hour1 + 134 x_user1_load3_hour2 + 131 x_user1_load3_hour3 
 + 121 x_user1_load3_hour4 + 112 x_user1_load3_hour5 + 100 x_user1_load3_hour6 
 + 93 x_user1_load3_hour7 + 118 x_user1_load4_hour0 + 107 x_user1_load4_hour1 
 + 102 x_user1_load4_hour2 + 98 x_user1_load4_hour3 + 94 x_user1_load4_hour4 
 + 83 x_user1_load4_hour5 + 74 x_user1_load4_hour6 + 72 x_user1_load4_hour7 
 + 66 x_user1_load4_hour8 + 159 x_user1_load5_hour0 + 151 x_user1_load5_hour1 
 + 136 x_user1_load5_hour2 + 134 x_user1_load5_hour3 + 127 x_user1_load5_hour4 
 + 119 x_user1_load5_hour5 + 102 x_user1_load5_hour6 + 96 x_user1_load5_hour7 
 + 93 x_user1_load5_hour8 

Subject To 
 load_load0_scheduled: + 1 x_user1_load0_hour0 + 1 x_user1_load0_hour1 
 + 1 x_user1_load0_hour2 + 1 x_user1_load0_hour3 + 1 x_user1_load0_hour4 
 + 1 x_user1_load0_hour5 + 1 x_user1_load0_hour6 + 1 x_user1_load0_hour7  = 1.0
 load_load1_scheduled: + 1 x_user1_load1_hour0 + 1 x_user1_load1_hour1 
 + 1 x_user1_load1_hour2 + 1 x_user1_load1_hour3 + 1 x_user1_load1_hour4 
 + 1 x_user1_load1_hour5 + 1 x_user1_load1_hour6 + 1 x_user1_load1_hour7 
 + 1 x_user1_load1_hour8  = 1.0
 load_load2_scheduled: + 1 x_user1_load2_hour0 + 1 x_user1_load2_hour1 
 + 1 x_user1_load2_hour2 + 1 x_user1_load2_hour3 + 1 x_user1_load2_hour4 
 + 1 x_user1_load2_hour5 + 1 x_user1_load2_hour6 + 1 x_user1_load2_hour7  = 1.0
 load_load3_scheduled: + 1 x_user1_load3_hour0 + 1 x_user1_load3_hour1 
 + 1 x_user1_load3_hour2 + 1 x_user1_load3_hour3 + 1 x_user1_load3_hour4 
 + 1 x_user1_load3_hour5 + 1 x_user1_load3_hour6 + 1 x_user1_load3_hour7  = 1.0
 load_load4_scheduled: + 1 x_user1_load4_hour0 + 1 x_user1_load4_hour1 
 + 1 x_user1_load4_hour2 + 1 x_user1_load4_hour3 + 1 x_user1_load4_hour4 
 + 1 x_user1_load4_hour5 + 1 x_user1_load4_hour6 + 1 x_user1_load4_hour7 
 + 1 x_user1_load4_hour8  = 1.0
 load_load5_scheduled: + 1 x_user1_load5_hour0 + 1 x_user1_load5_hour1 
 + 1 x_user1_load5_hour2 + 1 x_user1_load5_hour3 + 1 x_user1_load5_hour4 
 + 1 x_user1_load5_hour5 + 1 x_user1_load5_hour6 + 1 x_user1_load5_hour7 
 + 1 x_user1_load5_hour8  = 1.0
 MaxLoadConnection_user1_0: + 3 x_user1_load0_hour0 + 2 x_user1_load1_hour0 
 + 4 x_user1_load2_hour0 + 2 x_user1_load3_hour0 + 1 x_user1_load4_hour0 
 + 3 x_user1_load5_hour0  <= 4.0
 MaxLoadConnection_user1_1: + 3 x_user1_load0_hour1 + 2 x_user1_load0_hour0 
 + 2 x_user1_load1_hour1 + 4 x_user1_load1_hour0 + 4 x_user1_load2_hour1 
 + 1 x_user1_load2_hour0 + 2 x_user1_load3_hour1 + 1 x_user1_load3_hour0 
 + 1 x_user1_load4_hour1 + 2 x_user1_load4_hour0 + 3 x_user1_load5_hour1 
 + 1 x_user1_load5_hour0  <= 4.0
 MaxLoadConnection_user1_2: + 3 x_user1_load0_hour2 + 2 x_user1_load0_hour1 
 + 1 x_user1_load0_hour0 + 2 x_user1_load1_hour2 + 4 x_user1_load1_hour1 
 + 4 x_user1_load2_hour2 + 1 x_user1_load2_hour1 + 2 x_user1_load2_hour0 
 + 2 x_user1_load3_hour2 + 1 x_user1_load3_hour1 + 1 x_user1_load3_hour0 
 + 1 x_user1_load4_hour2 + 2 x_user1_load4_hour1 + 3 x_user1_load5_hour2 
 + 1 x_user1_load5_hour1  <= 4.0
 MaxLoadConnection_user1_3: + 3 x_user1_load0_hour3 + 2 x_user1_load0_hour2 
 + 1 x_user1_load0_hour1 + 2 x_user1_load1_hour3 + 4 x_user1_load1_hour2 
 + 4 x_user1_load2_hour3 + 1 x_user1_load2_hour2 + 2 x_user1_load2_hour1 
 + 2 x_user1_load3_hour3 + 1 x_user1_load3_hour2 + 1 x_user1_load3_hour1 
 + 1 x_user1_load4_hour3 + 2 x_user1_load4_hour2 + 3 x_user1_load5_hour3 
 + 1 x_user1_load5_hour2  <= 4.0
 MaxLoadConnection_user1_4: + 3 x_user1_load0_hour4 + 2 x_user1_load0_hour3 
 + 1 x_user1_load0_hour2 + 2 x_user1_load1_hour4 + 4 x_user1_load1_hour3 
 + 4 x_user1_load2_hour4 + 1 x_user1_load2_hour3 + 2 x_user1_load2_hour2 
 + 2 x_user1_load3_hour4 + 1 x_user1_load3_hour3 + 1 x_user1_load3_hour2 
 + 1 x_user1_load4_hour4 + 2 x_user1_load4_hour3 + 3 x_user1_load5_hour4 
 + 1 x_user1_load5_hour3  <= 4.0
 MaxLoadConnection_user1_5: + 3 x_user1_load0_hour5 + 2 x_user1_load0_hour4 
 + 1 x_user1_load0_hour3 + 2 x_user1_load1_hour5 + 4 x_user1_load1_hour4 
 + 4 x_user1_load2_hour5 + 1 x_user1_load2_hour4 + 2 x_user1_load2_hour3 
 + 2 x_user1_load3_hour5 + 1 x_user1_load3_hour4 + 1 x_user1_load3_hour3 
 + 1 x_user1_load4_hour5 + 2 x_user1_load4_hour4 + 3 x_user1_load5_hour5 
 + 1 x_user1_load5_hour4  <= 4.0
 MaxLoadConnection_user1_6: + 3 x_user1_load0_hour6 + 2 x_user1_load0_hour5 
 + 1 x_user1_load0_hour4 + 2 x_user1_load1_hour6 + 4 x_user1_load1_hour5 
 + 4 x_user1_load2_hour6 + 1 x_user1_load2_hour5 + 2 x_user1_load2_hour4 
 + 2 x_user1_load3_hour6 + 1 x_user1_load3_hour5 + 1 x_user1_load3_hour4 
 + 1 x_user1_load4_hour6 + 2 x_user1_load4_hour5 + 3 x_user1_load5_hour6 
 + 1 x_user1_load5_hour5  <= 4.0
 MaxLoadConnection_user1_7: + 3 x_user1_load0_hour7 + 2 x_user1_load0_hour6 
 + 1 x_user1_load0_hour5 + 2 x_user1_load1_hour7 + 4 x_user1_load1_hour6 
 + 4 x_user1_load2_hour7 + 1 x_user1_load2_hour6 + 2 x_user1_load2_hour5 
 + 2 x_user1_load3_hour7 + 1 x_user1_load3_hour6 + 1 x_user1_load3_hour5 
 + 1 x_user1_load4_hour7 + 2 x_user1_load4_hour6 + 3 x_user1_load5_hour7 
 + 1 x_user1_load5_hour6  <= 4.0
 MaxLoadConnection_user1_8: + 2 x_user1_load0_hour7 + 1 x_user1_load0_hour6 
 + 2 x_user1_load1_hour8 + 4 x_user1_load1_hour7 + 1 x_user1_load2_hour7 
 + 2 x_user1_load2_hour6 + 1 x_user1_load3_hour7 + 1 x_user1_load3_hour6 
 + 1 x_user1_load4_hour8 + 2 x_user1_load4_hour7 + 3 x_user1_load5_hour8 
 + 1 x_user1_load5_hour7  <= 4.0
 MaxLoadConnection_user1_9: + 1 x_user1_load0_hour7 + 4 x_user1_load1_hour8 
 + 2 x_user1_load2_hour7 + 1 x_user1_load3_hour7 + 2 x_user1_load4_hour8 
 + 1 x_user1_load5_hour8  <= 4.0

Bounds

Binary
 x_user1_load0_hour0 x_user1_load0_hour1 x_user1_load0_hour2
  x_user1_load0_hour3 x_user1_load0_hour4 x_user1_load0_hour5
  x_user1_load0_hour6 x_user1_load0_hour7 x_user1_load1_hour0
  x_user1_load1_hour1 x_user1_load1_hour2 x_user1_load1_hour3
  x_user1_load1_hour4 x_user1_load1_hour5 x_user1_load1_hour6
  x_user1_load1_hour7 x_user1_load1_hour8 x_user1_load2_hour0
  x_user1_load2_hour1 x_user1_load2_hour2 x_user1_load2_hour3
  x_user1_load2_hour4 x_user1_load2_hour5 x_user1_load2_hour6
  x_user1_load2_hour7 x_user1_load3_hour0 x_user1_load3_hour1
  x_user1_load3_hour2 x_user1_load3_hour3 x_user1_load3_hour4
  x_user1_load3_hour5 x_user1_load3_hour6 x_user1_load3_hour7
  x_user1_load4_hour0 x_user1_load4_hour1 x_user1_load4_hour2
  x_user1_load4_hour3 x_user1_load4_hour4 x_user1_load4_hour5
  x_user1_load4_hour6 x_user1_load4_hour7 x_user1_load4_hour8
  x_user1_load5_hour0 x_user1_load5_hour1 x_user1_load5_hour2
  x_user1_load5_hour3 x_user1_load5_hour4 x_user1_load5_hour5
  x_user1_load5_hour6 x_user1_load5_hour7 x_user1_load5_hour8
General

End