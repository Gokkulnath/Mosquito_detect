import os
from pathlib import Path

from sor4onnx import rename
from scc4onnx import order_conversion
from scs4onnx import shrinking
import onnx 
import onnx2tf
# graph = 

onnx_graph = rename(
  old_new=["onnx::", ""],
  onnx_graph=graph,
  mode="full",
  search_mode="prefix_match",
)


onnx_graph = rename(
  old_new=["onnx::", ""],
  input_onnx_file_path="fusionnet_180x320.onnx",
  output_onnx_file_path="fusionnet_180x320_renamed.onnx",
  mode="full",
  search_mode="prefix_match",
)





order_converted_graph = order_conversion(
    onnx_graph=graph,
    input_op_names_and_order_dims={"images": [0,2,3,1]},
    channel_change_inputs={"images": 1},
    non_verbose=True,
)


shrunk_graph, npy_file_paths = shrinking(
  onnx_graph=graph,
  mode='npy',
  non_verbose=True
)


onnx_graph = remove(
    remove_node_names=['output'],
    onnx_graph=graph,
)




!sor4onnx -if /content/best.onnx -of /content/best_cleaned.onnx --old_new "onnx::" ""  --mode full --search_mode prefix_match
!scc4onnx --input_onnx_file_path /content/best_cleaned.onnx --output_onnx_file_path /content/best_cleaned_1.onnx   \
--input_op_names_and_order_dims images [0,2,3,1] --channel_change_inputs images 1
!scs4onnx /content/best_cleaned_1.onnx /content/best_cleaned_2.onnx  --mode shrink
!snd4onnx --remove_node_names "output" --input_onnx_file_path /content/best_cleaned_2.onnx  --output_onnx_file_path /content/best_cleaned_pre_nms.onnx
!onnx2tf -i /content/best_cleaned_pre_nms.onnx -ois images:1,3,640,640  oiqt -qt per-tensor


onnx2tf.convert(
    input_onnx_file_path="model.onnx",
    output_folder_path="model.tf",
    copy_onnx_input_output_names_to_tflite=True,
    non_verbose=True,
)