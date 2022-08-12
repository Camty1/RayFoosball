file(REMOVE_RECURSE
  "libFNET.a"
  "libFNET.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/FNET.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
