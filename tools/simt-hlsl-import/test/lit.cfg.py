import os
import lit.formats

config.name = "simt-hlsl-import"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = ['.hlsl']
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.test_source_root

config.substitutions.append(('%simt-hlsl-import', '@SIMT_HLSL_IMPORT_TEST_EXECUTABLE@'))
config.substitutions.append(('%mlir-file-check', '@MLIR_FILE_CHECK@'))
