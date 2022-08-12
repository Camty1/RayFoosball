file(REMOVE_RECURSE
  "libIRremote.a"
  "libIRremote.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/IRremote.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
