# aqui testejare que el meu mètode retorni el IMD que espero mitjançant el mètode de generar un senyal amb 
# un IMD específic (primer SMPTE (quadràtic) i després CCIF (senar))
from generator import generate_signal_with_target_imd_QUADRATIC

target_imd_test = 5.0 # Target IMD in % (e.g., 5%) 
fs_test_func = 48000
f1_test_func = 60.0
f2_test_func = 7000.0
amp1_test_func = 0.8
amp2_test_func = 0.2

print(f"Attempting to generate signal with target IMD: {target_imd_test:.2f}%")

distorted_signal_result, final_alpha_result, actual_imd_result, iterations_result = generate_signal_with_target_imd_QUADRATIC(
    target_imd=target_imd_test,
    fs=fs_test_func, f1=f1_test_func, f2=f2_test_func,
    amp1=amp1_test_func, amp2=amp2_test_func,
    tolerance=0.05 # Smaller tolerance for more precise IMD
)
